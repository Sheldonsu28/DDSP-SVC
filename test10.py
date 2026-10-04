"""Diagonal position-dependent mass, for PyTorch 2.x.

Input: [batch, time, channels], float32/float64; use fixed dataset-level
normalization and consistent dt. This is an unforced dynamics hypothesis,
not a validated speech model. Run this file to execute analytic/gradient checks.
"""
import math
import torch
from torch import nn
from torch.func import grad, jvp, vmap

class SmoothResidualBlock(nn.Module):
    def __init__(self, width, residual_scale=0.1):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.SiLU(inplace=False)
        self.residual_scale = residual_scale

    def forward(self, x):
        h = self.fc1(self.activation(x))
        h = self.fc2(self.activation(h))
        return x + self.residual_scale * h


class SmoothEnergyHead(nn.Module):
    """Frame-local MLP: [..., dim] -> [..., out_dim].

    Fixed optional normalization is INSIDE the function being differentiated,
    so autodiff includes its chain-rule factors. Default is identity.
    Fit statistics on training data only, once before training; retain them
    in state_dict and keep fixed thereafter. No whitening or projection.
    """
    def __init__(self, dim, out_dim, width=256, blocks=2, output_gain=0.01,
                 input_mean=None, input_scale=None, min_scale=0.1):
        super().__init__()
        if min(dim, out_dim, width) < 1 or blocks < 0:
            raise ValueError("Dimensions must be positive and blocks >= 0")
        if min_scale <= 0 or not math.isfinite(min_scale):
            raise ValueError("min_scale must be finite and positive")
        mean = (torch.zeros(dim) if input_mean is None else
                torch.as_tensor(input_mean, dtype=torch.float32).detach().clone())
        scale = (torch.ones(dim) if input_scale is None else
                 torch.as_tensor(input_scale, dtype=torch.float32).detach().clone())
        if mean.shape != (dim,) or scale.shape != (dim,):
            raise ValueError("Normalization statistics must have shape [dim]")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("Normalization statistics must be finite")
        if (scale < 0).any():
            raise ValueError("input_scale must be nonnegative")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale.clamp_min(min_scale))
        self.input_layer = nn.Linear(dim, width)
        self.blocks = nn.ModuleList([
            SmoothResidualBlock(width) for _ in range(blocks)
        ])
        self.activation = nn.SiLU(inplace=False)
        # Potential offset has no effect on its gradient; mass-logit biases
        # are also unnecessary here because the hidden layers have biases.
        self.output_layer = nn.Linear(width, out_dim, bias=False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Small but nonzero: start near identity mass/weak potential forces,
        # while allowing gradients into the hidden layers immediately.
        nn.init.normal_(self.output_layer.weight,
                        std=output_gain / math.sqrt(width))

    def forward(self, q):
        h = self.input_layer((q - self.input_mean) / self.input_scale)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(self.activation(h))


def make_energy_heads(dim=768, mass_width=192, potential_width=256,
                      blocks=2, input_mean=None, input_scale=None):
    """Returns mass_net, potential_net; retain existing external constraints."""
    kwargs = dict(blocks=blocks, input_mean=input_mean, input_scale=input_scale)
    return (
        SmoothEnergyHead(dim, dim, width=mass_width, **kwargs),
        SmoothEnergyHead(dim, 1, width=potential_width, **kwargs),
    )

class DiagonalLagrangian(nn.Module):
    def __init__(self, dim, hidden=768, mass_range=0.5):
        super().__init__()
        if dim < 1 or hidden < 1 or not 0 < mass_range < 1:
            raise ValueError("Require dim, hidden >= 1 and 0 < mass_range < 1")
        self.dim = dim
        self.mass_range = mass_range
        self.mass_net, self.potential_net = make_energy_heads(
            dim=dim,
            mass_width=hidden,
            potential_width=hidden,
            blocks=2,
        )
        # Start with identity mass. Hidden mass-layer gradients start on
        # subsequent updates once the output layer becomes nonzero.
        # nn.init.zeros_(self.mass_net[-1].weight)
        # nn.init.zeros_(self.mass_net[-1].bias)
        
           
    def mass(self, q):
        """Positive diagonal entries; mean=1; each in (1-range,1+range).

        Mean normalization fixes the scale by convention. In dimension 1
        this fixes mass=1; this architecture is intended for vector latents.
        """
        h = torch.tanh(self.mass_net(q))
        return 1 + 0.5 * self.mass_range * (h - h.mean(dim=-1, keepdim=True))

    def potential(self, q):
        return self.potential_net(q).squeeze(-1)

    def kinetic(self, q, v):
        return 0.5 * (self.mass(q) * v.square()).sum(dim=-1)

    def residual_one(self, q, v, a):
        """One frame: q, v, a are vectors [D]. No dense Jacobian is built.

        r_i = m_i*a_i + (J_m @ v)_i*v_i
              - 0.5*sum_j (partial m_j/partial q_i)*v_j**2
              + partial V/partial q_i.

        grad(..., argnums=0) holds v fixed for the partial derivative;
        outer backprop still differentiates through q, v, and a.
        """
        m, dm_dt = jvp(self.mass, (q,), (v,))
        dT_dq = grad(self.kinetic, argnums=0)(q, v)
        dV_dq = grad(self.potential)(q)
        return m * a + dm_dt * v - dT_dq + dV_dq

    def forward(self, trajectory, dt=1.0, valid=None):
        """Return residuals [B,L-2,D].

        valid: optional bool [B,L]. Only stencils with all three frames
        valid contribute. Sanitize invalid padded frames before derivatives.
        Do not concatenate utterances without invalid boundary frames.
        Use ordinary finite padding and valid, or unpadded crops.
        """
        if trajectory.ndim != 3 or trajectory.shape[-1] != self.dim:
            raise ValueError("Expected trajectory [B,L,dim]")
        if trajectory.shape[1] < 3 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Require L>=3 and finite dt>0")
        q = trajectory
        if valid is not None:
            if valid.shape != q.shape[:2] or valid.dtype != torch.bool:
                raise ValueError("valid must be bool [B,L]")
            q = torch.where(valid[..., None], q, torch.zeros_like(q))
        position = q[:, 1:-1]
        velocity = (q[:, 2:] - q[:, :-2]) / (2 * dt)
        acceleration = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]) / dt**2
        shape = position.shape
        residual = vmap(self.residual_one)(
            position.reshape(-1, self.dim),
            velocity.reshape(-1, self.dim),
            acceleration.reshape(-1, self.dim),
        ).reshape(shape)
        return residual

    def loss(self, trajectory, dt=1.0, valid=None):
        residual = self(trajectory, dt=dt, valid=valid)
        per_frame = residual.square().mean(dim=-1)
        if valid is None:
            return per_frame.mean()
        interior = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
        count = interior.sum()
        if count.item() == 0:
            raise ValueError("No valid three-frame stencil")
        return per_frame.masked_select(interior).sum() / count


def self_test():
    """Analytic cross-coordinate check plus parameter/input backprop checks."""
    torch.manual_seed(7)

    class Analytic(DiagonalLagrangian):
        def mass(self, q):
            # Cross-coordinate dependence catches the incorrect 1D shortcut.
            return torch.stack((1 + q[1]**2, 2 + q[0]**2))

        def potential(self, q):
            return 0.5 * q.square().sum()

    analytic = Analytic(2).double()
    q, v, a = torch.randn(3, 2, dtype=torch.double)
    expected = torch.stack((
        (1 + q[1]**2)*a[0] + 2*q[1]*v[1]*v[0] - q[0]*v[1]**2 + q[0],
        (2 + q[0]**2)*a[1] + 2*q[0]*v[0]*v[1] - q[1]*v[0]**2 + q[1],
    ))
    torch.testing.assert_close(analytic.residual_one(q, v, a), expected)

    model = DiagonalLagrangian(3, hidden=8).double()
    # Exercise nonconstant mass, not just the identity initialization.
    nn.init.normal_(model.mass_net[-1].weight, std=0.1)
    nn.init.normal_(model.mass_net[-1].bias, std=0.1)
    sample = torch.randn(2, 6, 3, dtype=torch.double)
    model.loss(sample).backward()
    for net in (model.mass_net, model.potential_net):
        grads = [p.grad for p in net.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    prediction = sample.clone().requires_grad_(True)
    model.loss(prediction).backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    small = torch.randn(1, 4, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(model.loss, (small,), atol=1e-4, rtol=1e-3)

    # Padding is excluded, with a finite input gradient.
    valid = torch.tensor([[True, True, True, True, False, False]])
    padded = torch.randn(1, 6, 3, dtype=torch.double, requires_grad=True)
    torch.testing.assert_close(model.loss(padded, valid=valid), model.loss(padded[:, :4]))
    print("Passed: analytic residual, network gradients, frozen-model input "
          "gradients, numerical gradcheck, and padding mask.")

def acceleration_metrics(model, q, dt=1.0, valid=None, eps=1e-8):
    """
    q:     [B, L, D]
    valid: optional bool [B, L]; True for real frames.

    Evaluation/logging only: detaches q and returns detached metrics.
    """
    with torch.inference_mode(False), torch.enable_grad():
        q = q.detach().clone().float()
        valid = valid.clone() if valid is not None else None

        with torch.autocast(device_type=q.device.type, enabled=False):
            if valid is not None:
                q = torch.where(
                    valid[..., None], q, torch.zeros_like(q)
                )

            residual = model(q, dt=dt, valid=valid)
            mass = model.mass(q[:, 1:-1])

            acceleration = (
                q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]
            ) / dt**2

            # Average over latent coordinates first: [B, L-2]
            error = (residual / mass).square().mean(dim=-1)
            baseline = acceleration.square().mean(dim=-1)

            if valid is not None:
                mask = (
                    valid[:, :-2]
                    & valid[:, 1:-1]
                    & valid[:, 2:]
                )
                error = error.masked_select(mask)
                baseline = baseline.masked_select(mask)

            if error.numel() == 0:
                raise ValueError("No valid three-frame stencils")

            error_sum = error.sum().detach()
            baseline_sum = baseline.sum().detach()
            count = error.numel()

    error_mse = error_sum / count
    baseline_mse = baseline_sum / count
    normalized = error_mse / baseline_mse.clamp_min(eps)

    return {
        "acceleration_mse": error_mse,
        "baseline_mse": baseline_mse,
        "normalized_error": normalized,
        "improvement_percent": 100 * (1 - normalized),
        # Retain these for correct aggregation across unequal lengths.
        "error_sum": error_sum,
        "baseline_sum": baseline_sum,
        "frame_count": count,
    }
    
if __name__ == "__main__":
    self_test()