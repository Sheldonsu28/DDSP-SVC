import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from .msd import ScaleDiscriminator
from .mpd import MultiPeriodDiscriminator
from .mrd import  MultiResolutionDiscriminator
from torch.utils.checkpoint import checkpoint


def _sub_disc_losses(d, fake, real):
    # Runs inside checkpoint and returns scalars only, so no conv activation
    # is a checkpoint output: all of them are freed after forward and
    # recomputed on backward. Returning fmap tensors instead would keep every
    # intermediate activation alive and defeat the checkpointing.
    x = torch.cat([fake, real], dim=0)
    out = d(x)
    if isinstance(out, list):  # ScaleDiscriminator returns [(fmap, score)]
        out = out[0]
    fmap, score = out
    b = fake.shape[0]
    score_f, score_r = score[:b], score[b:]
    feat = fake.new_zeros(())
    for fm in fmap:
        feat = feat + F.l1_loss(fm[:b], fm[b:].detach())
    score_term = torch.mean((score_f - 1.0) ** 2)
    d_term = torch.mean((score_r - 1.0) ** 2) + torch.mean(score_f ** 2)
    return score_term, feat, d_term


def _checkpointed_losses(subs, fake, real):
    score_loss = 0.0
    feat_loss = 0.0
    loss_d = 0.0
    for d in subs:
        s, f, dl = _sub_disc_losses(d, fake, real)
        score_loss = score_loss + s
        feat_loss = feat_loss + f
        loss_d = loss_d + dl
    n = len(subs)
    return score_loss / n, (feat_loss / n) * 2, loss_d / n


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)
        self.MSD = ScaleDiscriminator()

    def forward(self, x):
        r = self.MRD(x)
        p = self.MPD(x)
        s = self.MSD(x)
        return r + p + s

    def compute_losses(self, fake, real):
        """LS-GAN losses over all sub-discriminators in one batched fake+real
        pass per sub-discriminator: (score_loss, feat_loss, loss_d), scaled
        identically to the inline loops in train_vc_osgan (mean over subs,
        feature-matching x2, real features detached)."""
        subs = list(self.MRD.discriminators) + list(self.MPD.discriminators) + [self.MSD]
        return _checkpointed_losses(subs, fake, real)


class Discriminator2(nn.Module):
    def __init__(self, hp):
        super(Discriminator2, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)

    def forward(self, x):
        r = checkpoint(self.MRD, x, use_reentrant=False)
        p = checkpoint(self.MPD, x, use_reentrant=False)
        return r + p

    def compute_losses(self, fake, real):
        subs = list(self.MRD.discriminators) + list(self.MPD.discriminators)
        return _checkpointed_losses(subs, fake, real)
