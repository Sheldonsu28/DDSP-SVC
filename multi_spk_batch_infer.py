import argparse
import os
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from tqdm import tqdm, TqdmExperimentalWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Reuse existing inference pipeline
import batch_infer as bi
from ddsp.vocoder import Units_Encoder
from reflow.vocoder import load_model_vocoder


# -----------------------------
# CLI and config helpers
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch infer by parsing multiple speakers from filenames and running per-speaker outputs"
        )
    )
    p.add_argument("-m", "--model_ckpt", required=True, type=str, help="Path to model .pt")
    p.add_argument("-i", "--input", required=True, type=str, help="Input directory of dry vocals")
    p.add_argument("-o", "--output", required=True, type=str, help="Output directory root")
    p.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="cpu or cuda (auto if not set)",
    )
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of processes for inference. GPU models often prefer small values (1-2)."
        ),
    )
    p.add_argument(
        "-e",
        "--extensions",
        nargs="*",
        default=["wav", "flac"],
        help="Audio file extensions to include",
    )

    # Forwardable params to batch_infer.infer
    p.add_argument("-pe", "--pitch_extractor", type=str, default="rmvpe")
    p.add_argument("-fmin", "--f0_min", type=str, default="50")
    p.add_argument("-fmax", "--f0_max", type=str, default="1100")
    p.add_argument("-k", "--key", type=str, default="0")
    p.add_argument("-f", "--formant_shift_key", type=str, default="0")
    p.add_argument("-v", "--vocal_register_shift_key", type=str, default="0")
    p.add_argument("-th", "--threhold", type=str, default="-60")
    p.add_argument("-step", "--infer_step", type=str, default="200")
    p.add_argument("-method", "--method", type=str, default="euler")
    p.add_argument("-ts", "--t_start", type=str, default="0.0")

    p.add_argument(
        "--skip_no_match",
        action="store_true",
        help="Skip files where no speaker is matched; otherwise logs a warning",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-worker child progress bars (multiprocess-safe)",
    )

    return p.parse_args()


# -----------------------------
# Speaker parsing
# -----------------------------


def list_audio_files(root: pathlib.Path, extensions: Sequence[str]) -> List[pathlib.Path]:
    exts = {e.lower().lstrip(".") for e in extensions}
    files: List[pathlib.Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    files.sort()
    return files


def extract_trailing_segment(filename_stem: str) -> str:
    # Split only on the first underscore: songname_speakerlist
    if "_" in filename_stem:
        return filename_stem.split("_", 1)[1]
    return filename_stem


_SEP_RE = re.compile(r"[\s_\-\|,，、\.。:：;；!！\?？\(\)（）\[\]【】{}《》<>\"“”‘’&\+·…—–]+")


def _normalize_token(s: str) -> str:
    # Remove common separators/punctuations and digits; keep CJK letters as-is
    s = _SEP_RE.sub("", s)
    s = re.sub(r"\d+", "", s)
    return s


def find_speakers_in_name(trailing: str, spk_names: Sequence[str]) -> List[str]:
    # Normalize once
    trailing_norm = _normalize_token(trailing)
    tokens = [t for t in _SEP_RE.split(trailing) if t]
    token_norms = [_normalize_token(t) for t in tokens]

    # Build normalized dictionary
    norm_map: Dict[str, str] = {}
    for s in spk_names:
        if not s:
            continue
        norm = _normalize_token(s)
        if norm:
            norm_map[norm] = s

    # Find occurrences in normalized trailing, plus token-in-speaker fallback
    found: List[Tuple[int, int, str]] = []  # (idx, -len, original)
    for norm, orig in norm_map.items():
        idx = trailing_norm.find(norm)
        matched = False
        if idx >= 0:
            matched = True
        else:
            # token subset matches (e.g., '鬼谷子' matches '鬼谷子（混响大）')
            for tok in token_norms:
                if len(tok) >= 2 and norm.find(tok) >= 0:
                    idx = max(trailing_norm.find(tok), 0)
                    matched = True
                    break
        if matched:
            found.append((idx, -len(norm), orig))

    if not found:
        return []

    # Sort by position then prefer longer matches
    found.sort()
    res: List[str] = []
    seen = set()
    for _, _, orig in found:
        if orig not in seen:
            res.append(orig)
            seen.add(orig)
    return res


# -----------------------------
# Worker process state and task
# -----------------------------


_G = {
    "device": None,
    "model": None,
    "vocoder": None,
    "args": None,
    "units_encoder": None,
    "tqdm_position": 0,
}


@dataclass
class CmdLike:
    # Only the fields used inside batch_infer.infer
    pitch_extractor: str
    f0_min: str
    f0_max: str
    key: str
    formant_shift_key: str
    vocal_register_shift_key: str
    threhold: str
    spk_mix_dict: str
    spk_id: str
    infer_step: str
    method: str
    t_start: str


def _init_worker(model_ckpt: str, device: Optional[str], encoder_name: str, encoder_ckpt: str,
                 encoder_sr: int, encoder_hop: int, cnhubertsoft_gate: int,
                 tqdm_lock, pos_queue, verbose: bool) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _G["device"] = device
    # Configure tqdm multi-process lock
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.set_lock(tqdm_lock)
    except Exception:
        pass
    model, vocoder, args = load_model_vocoder(model_ckpt, device=device)
    # Set up units encoder consistent with config
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
        encoder_name,
        encoder_ckpt,
        encoder_sr,
        encoder_hop,
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device,
    )
    _G["model"] = model
    _G["vocoder"] = vocoder
    _G["args"] = args
    _G["units_encoder"] = units_encoder
    # Monkeypatch tqdm in reflow to position per-worker or disable
    try:
        import reflow.reflow as reflow_mod
        from tqdm import tqdm as orig_tqdm
        if verbose:
            # Assign a persistent position slot per worker
            pos = pos_queue.get()
            _G["tqdm_position"] = pos

            def _wrapped_tqdm(iterable=None, *a, **k):
                if "position" not in k:
                    k["position"] = _G["tqdm_position"]
                if "leave" not in k:
                    k["leave"] = False
                # Prefix desc with worker tag for clarity
                if "desc" in k and k["desc"]:
                    k["desc"] = f"w{_G['tqdm_position']} | {k['desc']}"
                return orig_tqdm(iterable, *a, **k)

            reflow_mod.tqdm = _wrapped_tqdm  # type: ignore
        else:
            # No child bars; keep console clean
            def _noop_tqdm(iterable=None, *a, **k):
                return iterable if iterable is not None else range(int(k.get("total", 0) or 0))

            reflow_mod.tqdm = _noop_tqdm  # type: ignore
    except Exception:
        pass


def _run_one(input_path: str, output_path: str, cmd_dict: Dict[str, str]) -> Tuple[str, bool, str]:
    t0 = time.time()
    try:
        cmd = CmdLike(
            pitch_extractor=cmd_dict["pitch_extractor"],
            f0_min=cmd_dict["f0_min"],
            f0_max=cmd_dict["f0_max"],
            key=cmd_dict["key"],
            formant_shift_key=cmd_dict["formant_shift_key"],
            vocal_register_shift_key=cmd_dict["vocal_register_shift_key"],
            threhold=cmd_dict["threhold"],
            spk_mix_dict="None",
            spk_id=cmd_dict["spk_id"],
            infer_step=cmd_dict["infer_step"],
            method=cmd_dict["method"],
            t_start=cmd_dict["t_start"],
        )
        # print(cmd_dict)
        cmd = f"F:/AI/DDSP-SVC/.venv/Scripts/python.exe main_reflow.py -i {input_path} -m {cmd_dict['model_path']} -o {output_path} -k 0 -id {cmd_dict['spk_id']} -method euler -step 200 -pe rmvpe -ts 0.0 -v 0 -f 0"
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        # bi.infer(
        #     input_path,
        #     output_path,
        #     cmd,
        #     _G["device"],
        #     _G["model"],
        #     _G["vocoder"],
        #     _G["args"],
        #     _G["units_encoder"],
        # )
        dt = (time.time() - t0) * 1000.0
        return output_path, True, f"OK ({dt:.0f} ms)"
    except Exception as e:
        return output_path, False, f"ERROR: {e}"


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    cmd = parse_args()

    # Avoid touching CUDA in parent; just parse config.yaml next to checkpoint
    config_path = os.path.join(os.path.split(cmd.model_ckpt)[0], 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    encoder_name = data_cfg.get('encoder')
    encoder_ckpt = data_cfg.get('encoder_ckpt')
    encoder_sr = data_cfg.get('encoder_sample_rate')
    encoder_hop = data_cfg.get('encoder_hop_size')
    cnhubertsoft_gate = data_cfg.get('cnhubertsoft_gate', 10)

    # Build mapping speaker name -> 1-based ID from config
    spk_names: List[str] = list((cfg.get('spks') or []))
    offset = cfg.get('start')
    name_to_id: Dict[str, int] = {name: i for i, name in enumerate(spk_names,start=offset)}

    in_root = pathlib.Path(cmd.input)
    out_root = pathlib.Path(cmd.output)
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(in_root, cmd.extensions)
    if not files:
        print(f"No input files found under: {in_root}")
        return

    # Build tasks by parsing speakers from filenames
    tasks: List[Tuple[str, str, Dict[str, str]]] = []
    for f in files:
        stem = f.stem
        trailing = extract_trailing_segment(stem)
        matched_spks = find_speakers_in_name(trailing, spk_names)
        if not matched_spks:
            msg = f"No speakers matched for file: {f.name} (segment='{trailing}')"
            if cmd.skip_no_match:
                print("[Skip] " + msg)
                continue
            else:
                print("[Warn] " + msg)
                continue

        # Schedule per-speaker outputs under out_root/<spk>/relative_path.wav
        rel_path = f.relative_to(in_root)
        for spk in matched_spks:
            spk_id = name_to_id.get(spk)
            if not spk_id:
                print(f"[Warn] Speaker '{spk}' not in config spks; skipping {f.name}")
                continue
            out_dir = out_root / spk / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = (out_dir / rel_path.name).with_suffix(".wav")
            params = {
                "pitch_extractor": cmd.pitch_extractor,
                "f0_min": str(cmd.f0_min),
                "f0_max": str(cmd.f0_max),
                "key": str(cmd.key),
                "formant_shift_key": str(cmd.formant_shift_key),
                "vocal_register_shift_key": str(cmd.vocal_register_shift_key),
                "threhold": str(cmd.threhold),
                "infer_step": str(cmd.infer_step),
                "method": str(cmd.method),
                "t_start": str(cmd.t_start),
                "spk_id": str(spk_id),
                "model_path":cmd.model_ckpt
            }
            tasks.append((str(f), str(out_path), params))

    if not tasks:
        print("No tasks to run.")
        return

    print(f"Found {len(files)} files; scheduled {len(tasks)} speaker-specific tasks.")
    print(f"Device: {cmd.device or 'auto'} | Workers: {cmd.workers}")

    # Run tasks with processes. Each process initializes model + encoder once.
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    # Shared lock for tqdm cross-process output (use same context as pool)
    tqdm_lock = ctx.RLock()
    # Pre-allocate worker positions 1..workers for child bars (0 reserved for parent bar)
    pos_queue = ctx.Queue()
    for pos in range(1, max(1, int(cmd.workers)) + 1):
        pos_queue.put(pos)

    with ProcessPoolExecutor(
        max_workers=max(1, int(cmd.workers)),
        mp_context=ctx,
        # initializer=_init_worker,
        # initargs=(
        #     cmd.model_ckpt,
        #     cmd.device,
        #     encoder_name,
        #     encoder_ckpt,
        #     encoder_sr,
        #     encoder_hop,
        #     cnhubertsoft_gate,
        #     tqdm_lock,
        #     pos_queue,
        #     bool(cmd.verbose),
        # ),
    ) as ex:
        futs = [ex.submit(_run_one, ip, op, p) for ip, op, p in tasks]
        done_cnt = 0
        # Parent-level tasks bar at position 0
        tqdm.set_lock(tqdm_lock)
        with tqdm(total=len(tasks), desc="tasks", dynamic_ncols=True, position=0) as pbar:
            for fut in as_completed(futs):
                out_path, ok, msg = fut.result()
                done_cnt += 1
                status = "OK" if ok else "FAIL"
                tqdm.write(f"[{status}] ({done_cnt}/{len(tasks)}) -> {out_path} | {msg}")
                pbar.update(1)


if __name__ == "__main__":
    main()
