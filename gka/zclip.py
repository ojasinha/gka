"""ZClip: adaptive gradient clipping by z-score of the gradient norm (Kumar et al., arXiv:2504.02507).

Keeps an exponential moving average of the mean and variance of the total gradient norm. When the
current norm's z-score z = (g - mean) / std exceeds z_thresh, the gradients are rescaled to

    threshold = mean + z_thresh**2 / z * std      ("adaptive scaling": the bigger the spike, the harder the clip)

and the statistics are updated with the post-clip norm, so a spike does not inflate them. The first
warmup_steps norms only initialise the statistics. max_norm is a fixed ceiling applied throughout
(during warmup it is the only clip), the same 1.0 clip the earlier runs used.
"""

import math

import torch


class ZClip:
    def __init__(self, alpha=0.97, z_thresh=2.5, warmup_steps=25, max_norm=1.0, eps=1e-6):
        self.alpha, self.z_thresh, self.warmup_steps, self.max_norm, self.eps = alpha, z_thresh, warmup_steps, max_norm, eps
        self.buffer = []  # norms seen during warmup
        self.mean = None
        self.var = None
        self.n_clipped = 0  # z-score clips (excludes the fixed max_norm ceiling)

    def threshold(self, g: float):
        """Clip target for a raw norm g, or None if no z-score clip applies."""
        if self.mean is None:
            return None
        z = (g - self.mean) / (math.sqrt(self.var) + self.eps)
        if z <= self.z_thresh:
            return None
        return self.mean + self.z_thresh**2 / z * math.sqrt(self.var)

    def _update(self, g: float):
        self.mean = self.alpha * self.mean + (1 - self.alpha) * g
        self.var = self.alpha * self.var + (1 - self.alpha) * (g - self.mean) ** 2

    @torch.no_grad()
    def step(self, params) -> dict:
        """Clip the gradients of params in place. Returns the raw and post-clip norms."""
        params = [p for p in params if p.grad is not None]
        g = torch.nn.utils.get_total_norm([p.grad for p in params]).item()
        warming_up = self.mean is None
        target, zclipped = g, False
        if not warming_up:
            t = self.threshold(g)
            if t is not None:
                target, zclipped = t, True
                self.n_clipped += 1
        if self.max_norm is not None:
            target = min(target, self.max_norm)
        target = min(target, g)
        if target < g:
            torch.nn.utils.clip_grads_with_norm_(params, target, torch.tensor(g, device=params[0].grad.device))
        if warming_up:  # collect raw norms; initialise the statistics once the buffer is full
            self.buffer.append(g)
            if len(self.buffer) == self.warmup_steps:
                n = len(self.buffer)
                self.mean = sum(self.buffer) / n
                self.var = sum((x - self.mean) ** 2 for x in self.buffer) / n
                self.buffer = []
        else:  # EMA update with the post-clip norm, so spikes don't inflate the statistics
            self._update(target)
        return {"grad_norm": g, "clipped_norm": target, "zclipped": zclipped}

    def state_dict(self):
        return {"buffer": list(self.buffer), "mean": self.mean, "var": self.var, "n_clipped": self.n_clipped}

    def load_state_dict(self, s):
        self.buffer, self.mean, self.var, self.n_clipped = list(s["buffer"]), s["mean"], s["var"], s["n_clipped"]
