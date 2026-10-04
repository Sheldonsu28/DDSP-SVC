import torch
import torch.nn as nn

from stylizer.modules_grl import SpeakerClassifier
from stylizer.resblocks import PosteriorEncoder, ResidualCouplingBlock, TextEncoder
import commons
import torch.nn.functional as F

class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(256, 256)
        self.enc_p = TextEncoder(
            1280,
            768,
            768,
            192,
            640,
            2,
            6,
            3,
            0.1,
        )
        self.speaker_classifier = SpeakerClassifier(
            192,
            256,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            192,
            192,
            5,
            1,
            16,
            gin_channels=256,
        )
        self.flow = ResidualCouplingBlock(
            192,
            192,
            5,
            1,
            4,
            gin_channels=256
        )

    def forward(self, ppg, vec, pit, spec, spk, ppg_l, spec_l):
        ppg = ppg + torch.randn_like(ppg) * 1  # Perturbation
        vec = vec + torch.randn_like(vec) * 2  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec)
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True)
        # speaker
        spk_preds = self.speaker_classifier(x)
        return spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds

    def infer(self, ppg, vec, pit, spk, ppg_l):
        ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec)
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        return z * ppg_mask