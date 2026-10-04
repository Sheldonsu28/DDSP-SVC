"""
Training step for the frontend pre-processor.

Assembles all loss terms with proper weighting and warmup schedules.

This module assumes:
  - `frontend`        : LeakageReductionFrontend
  - `ddsp`            : frozen pre-trained DDSP-SVC model
  - `discriminator`   : multi-period + multi-resolution discriminator
  - `mel_extractor`   : function(audio) -> mel (B, 128, T_mel)
  - Batch is produced by the augmentation pipeline and contains:
        whisper, contentvec, hubert      : SSL features from AUGMENTED audio
        spk_emb                          : 192-d from speaker verification model
        f0, volume                       : from ORIGINAL target audio
                                           (f0 in LINEAR Hz; unvoiced = 0)
        spk_id                           : 1 or 2 (DDSP's internal speaker id)
        target_audio                     : original (un-augmented) waveform
        target_mel                       : mel of the original audio
        formant_target                   : supervised formant shift target
        aug_bucket                       : integer bucket ID for adversarial loss
"""

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Loss schedule helpers
# ----------------------------------------------------------------------

def warmup(step: int, start: int, end: int, max_value: float) -> float:
    """Linear warmup from 0 at `start` to `max_value` at `end`."""
    if step < start:
        return 0.0
    if step >= end:
        return max_value
    return max_value * (step - start) / (end - start)


def cooldown(step: int, start: int, end: int, max_value: float) -> float:
    """Linear cooldown from `max_value` at `start` to 0 at `end`."""
    if step < start:
        return max_value
    if step >= end:
        return 0.0
    return max_value * (1 - (step - start) / (end - start))


# ----------------------------------------------------------------------
# Generator (frontend + frozen DDSP) step
# ----------------------------------------------------------------------

def frontend_step(
    frontend, ddsp, discriminator, mel_extractor,
    batch, step: int,
    optimizer_G,
):
    """One training step for the generator side (frontend + frozen DDSP)."""

    # ---- Loss weight schedule ----
    w_mel      = 45.0
    w_adv_aud  = warmup(step, 1_000, 5_000, 1.0)     # delay adv to let mel settle
    w_fm       = warmup(step, 1_000, 5_000, 2.0)
    w_f0       = 0.5
    w_vol      = 0.3                                  # aux volume head
    w_adv_spk  = warmup(step, 5_000, 20_000, 0.1)    # GRL coefficient (also lambda inside model)
    w_fmt_sup  = cooldown(step, 20_000, 80_000, 0.1) # supervised formant target
    w_fmt_l2   = 0.01
    w_fmt_tv   = 0.05
    w_gate_l1  = 0.001

    # ---- Input noise warmup ----
    # Start with zero noise so the model finds a clean baseline, then ramp
    # up over steps 2K-15K to the target noise levels. The frontend reads
    # these attributes inside its forward(), so we just set them here.
    noise_ramp = warmup(step, 2_000, 15_000, 1.0)
    frontend.noise_whisper    = 0.10 * noise_ramp
    frontend.noise_contentvec = 0.15 * noise_ramp
    frontend.noise_hubert     = 0.10 * noise_ramp

    # GRL lambda follows w_adv_spk (we apply it inside the model)
    grl_lambda = float(w_adv_spk * 10)  # rescale because outer weight is small

    # ---- Forward through frontend ----
    cv_out, formant_shift, aux = frontend(
        whisper=batch['whisper'],
        contentvec=batch['contentvec'],
        hubert=batch['hubert'],
        spk_emb=batch['spk_emb'],
        grl_lambda=grl_lambda,
        return_aux=True,
    )

    # ---- Forward through frozen DDSP ----
    # NOTE: the exact signature depends on your DDSP wrapper. The pattern:
    #   DDSP consumes `cv_out` as its `units` input,
    #   plus F0, volume, spk_id, and `formant_shift` as `aug_shift`.
    audio_pred = ddsp(
        units=cv_out,
        f0=batch['f0'],
        volume=batch['volume'],
        spk_id=batch['spk_id'],
        aug_shift=formant_shift,   # (B, T, 1)
    )

    # ---- Mel reconstruction ----
    mel_pred = mel_extractor(audio_pred)
    mel_target = batch['target_mel']
    # Match lengths defensively
    T = min(mel_pred.size(-1), mel_target.size(-1))
    loss_mel = F.l1_loss(mel_pred[..., :T], mel_target[..., :T])

    # ---- Adversarial (generator side) + feature matching ----
    # Discriminator returns: list of (logit, [feature_maps]) tuples for each sub-D
    d_fake = discriminator(audio_pred)
    d_real = discriminator(batch['target_audio'])

    loss_adv = 0.0
    loss_fm = 0.0
    for (logit_f, feats_f), (_, feats_r) in zip(d_fake, d_real):
        loss_adv = loss_adv + torch.mean((logit_f - 1.0) ** 2)  # LSGAN
        for ff, fr in zip(feats_f, feats_r):
            loss_fm = loss_fm + F.l1_loss(ff, fr.detach())
    loss_adv = loss_adv / max(len(d_fake), 1)
    loss_fm = loss_fm / max(len(d_fake), 1)

    # ---- Auxiliary F0 ----
    # The model now outputs F0 in linear Hz (> 0 via softplus).
    # The batch target is also in linear Hz.
    # We compute L1 in log space for perceptual weighting -- an octave error
    # should weigh the same whether it occurs at 100 Hz or 800 Hz.
    # Unvoiced frames (f0 == 0 or below a floor) are masked out.
    f0_linear = batch['f0']                         # (B, T) or (B, T, 1) in Hz
    if f0_linear.dim() == 3:
        f0_linear = f0_linear.squeeze(-1)
    f0_pred = aux['f0_pred'].squeeze(-1)            # (B, T) in Hz, > 0
    voiced_mask = (f0_linear > 10.0).float()        # voiced if > 10 Hz
    # Match lengths defensively
    T_f0 = min(f0_pred.size(-1), f0_linear.size(-1))
    f0_pred = f0_pred[..., :T_f0]
    f0_linear = f0_linear[..., :T_f0]
    voiced_mask = voiced_mask[..., :T_f0]
    log_f0_target = torch.log(torch.clamp(f0_linear, min=10.0))
    log_f0_pred = torch.log(torch.clamp(f0_pred, min=10.0))
    # Masked L1 in log space
    loss_f0 = ((log_f0_pred - log_f0_target).abs() * voiced_mask).sum() \
              / (voiced_mask.sum() + 1e-6)

    # ---- Auxiliary Volume ----
    # batch['volume'] is windowed RMS amplitude (linear, not dB), produced by
    # the Volume_Extractor with formula sqrt(E[x^2] - E[x]^2). Range is
    # roughly 0.0 (silence) to ~0.3 (loud), with most voiced frames around
    # 0.05. Compute L1 in log space for dB-style perceptual weighting.
    #
    # The epsilon acts as a noise floor: 1e-3 is roughly -60 dB. Using a
    # larger eps than for F0 prevents silent frames (volume == 0) from
    # dominating the loss with huge log-space errors.
    vol_target = batch['volume']                    # (B, T) or (B, T, 1)
    if vol_target.dim() == 3:
        vol_target = vol_target.squeeze(-1)
    vol_pred = aux['vol_pred'].squeeze(-1)          # (B, T), > 0
    T_v = min(vol_pred.size(-1), vol_target.size(-1))
    vol_pred = vol_pred[..., :T_v]
    vol_target = vol_target[..., :T_v]
    eps = 1e-3
    log_vol_pred = torch.log(vol_pred + eps)
    log_vol_target = torch.log(torch.clamp(vol_target, min=0.0) + eps)
    loss_vol = (log_vol_pred - log_vol_target).abs().mean()

    # ---- Adversarial speaker (GRL applied inside model already) ----
    # We compute the standard CE; gradient is reversed inside the frontend.
    loss_adv_spk = F.cross_entropy(aux['adv_logits'], batch['aug_bucket'])

    # ---- Supervised formant shift (early-training only) ----
    # formant_target is the desired output for the formant_shift head
    # (e.g., -log2(formant_ratio) for the augmented samples).
    # Shape: (B,) scalar per sample; expand over time as a constant target.
    fmt_target = batch['formant_target'].view(-1, 1, 1).expand_as(formant_shift)
    loss_fmt_sup = F.l1_loss(formant_shift, fmt_target)

    # ---- Formant regularization ----
    loss_fmt_l2 = formant_shift.pow(2).mean()
    loss_fmt_tv = (formant_shift[:, 1:] - formant_shift[:, :-1]).abs().mean()

    # ---- Gate regularization (encourage closed gate by default) ----
    loss_gate_l1 = aux['cv_gate'].mean()

    # ---- Total loss ----
    loss_total = (
        w_mel * loss_mel
        + w_adv_aud * loss_adv
        + w_fm * loss_fm
        + w_f0 * loss_f0
        + w_vol * loss_vol
        + w_adv_spk * loss_adv_spk
        + w_fmt_sup * loss_fmt_sup
        + w_fmt_l2 * loss_fmt_l2
        + w_fmt_tv * loss_fmt_tv
        + w_gate_l1 * loss_gate_l1
    )

    optimizer_G.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(frontend.parameters(), max_norm=5.0)
    optimizer_G.step()

    return {
        'loss/total': loss_total.item(),
        'loss/mel': loss_mel.item(),
        'loss/adv': loss_adv.item(),
        'loss/fm': loss_fm.item(),
        'loss/f0': loss_f0.item(),
        'loss/vol': loss_vol.item(),
        'loss/adv_spk': loss_adv_spk.item(),
        'loss/fmt_sup': loss_fmt_sup.item(),
        'loss/fmt_l2': loss_fmt_l2.item(),
        'loss/fmt_tv': loss_fmt_tv.item(),
        'loss/gate_l1': loss_gate_l1.item(),
        'diag/fmt_mean': formant_shift.mean().item(),
        'diag/fmt_std': formant_shift.std().item(),
        'diag/gate_mean': aux['cv_gate'].mean().item(),
        'diag/gate_max': aux['cv_gate'].max().item(),
        'diag/f0_pred_mean_hz': (f0_pred * voiced_mask).sum().item()
                                / (voiced_mask.sum().item() + 1e-6),
        'diag/vol_pred_mean': vol_pred.mean().item(),
        'w/adv_aud': w_adv_aud, 'w/adv_spk': w_adv_spk, 'w/fmt_sup': w_fmt_sup,
    }


# ----------------------------------------------------------------------
# Discriminator step (standard)
# ----------------------------------------------------------------------

def discriminator_step(discriminator, audio_real, audio_fake, optimizer_D):
    """Standard LSGAN discriminator update."""
    audio_fake = audio_fake.detach()
    d_real = discriminator(audio_real)
    d_fake = discriminator(audio_fake)
    loss_d = 0.0
    for (logit_r, _), (logit_f, _) in zip(d_real, d_fake):
        loss_d = loss_d + torch.mean((logit_r - 1.0) ** 2) + torch.mean(logit_f ** 2)
    loss_d = loss_d / max(len(d_real), 1)

    optimizer_D.zero_grad()
    loss_d.backward()
    optimizer_D.step()
    return {'loss/D': loss_d.item()}