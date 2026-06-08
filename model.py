import torch
import torch.nn as nn


def _trunk(state_dim: int, hidden: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(state_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.5)
            nn.init.zeros_(m.bias)
    return net


class FlatActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = _trunk(state_dim, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        h = self.trunk(s)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, s, mask=None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v

    def evaluate(self, s, a, mask=None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), v


class AttentiveActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = _trunk(state_dim, hidden)
        self.state_proj = nn.Linear(hidden, hidden)
        self.word_embed = nn.Embedding(action_dim, hidden)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self._scale = hidden ** 0.5
        nn.init.normal_(self.word_embed.weight, 0, 0.01)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        h = self.trunk(s)
        logits = torch.matmul(self.state_proj(h), self.word_embed.weight.T) / self._scale
        return logits, self.critic(h).squeeze(-1)

    def act(self, s, mask=None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v

    def evaluate(self, s, a, mask=None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), v


def build_model(cfg, state_dim: int, action_dim: int):
    cls = AttentiveActorCritic if cfg.use_attention else FlatActorCritic
    return cls(state_dim, action_dim, cfg.hidden)


