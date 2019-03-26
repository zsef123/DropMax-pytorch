import torch
import torch.nn.functional as F
from torch import distributions


class DropMax:
    def __init__(self, device, tau=0.5, eps=1e-8, log_eps=1e-20):
        self.device = device
        self.tau = tau
        self.eps = eps
        self.log_eps = log_eps

    def _log(self, x):
        return torch.log(x + self.log_eps)

    def _logit(self, x):
        return self._log(x) - self._log(1 - x)

    def _cross_entropy(self, z, o, y):
        # expo = ((z if training else p) + eps)*exp(o)
        expo = (z + self.eps) * torch.exp(o)
        denom = self._log(expo.sum(dim=1))
        numer = self._log((expo * y).sum(dim=1))
        return -(numer - denom).mean()

    def _binary_mask(self, p, y):
        u = distributions.Uniform(torch.zeros_like(p), torch.ones_like(p)).sample()
        z = torch.sigmoid((self._logit(p) + self._logit(u)) / self.tau)
        return torch.where(y==1, torch.ones_like(z).to(self.device), z)

    def _kl_divergence(self, p, q, y):
        log_p, log_q = self._log(p), self._log(q)
        nontarget =  q * (log_q - log_p) + (1 - q) * (self._log(1 - q) - self._log(1 - p))
        kl = torch.where(y==1, -log_p, nontarget)
        return kl.mean(dim=0).sum()

    def _aux(self, r, y):
        target = -self._log(r)
        nontarget = -self._log(1 - r)
        aux = torch.where(y==1, target, nontarget)
        return aux.mean(dim=0).sum()

    def _entropy(self, p):
        ent = p * self._log(p) + (1 - p) * self._log(1 - p)
        return ent.mean(dim=0).sum()

    def get_acc(self, p, o, label):
        pred = (p + self.eps) * torch.exp(o)
        _, idx = pred.max(1)
        return (idx.cpu() == label).sum().item()

    def __call__(self, o, p, r, q, y):
        # net['cent'] + net['wd'] + net['kl'] + net['aux'] + net['ent']
        z = self._binary_mask(p, y)
        cent = self._cross_entropy(z, o, y)
        kl = self._kl_divergence(p, q, y)
        aux = self._aux(r, y)
        ent = self._entropy(p)
        return cent + kl + aux + ent
