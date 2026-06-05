import torch
import torch.nn as nn
import numpy as np


class WordAdversary(nn.Module):
    def __init__(self, n_answers: int, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_answers, hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.normal_(self.embed.weight, 0, 0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(idx)).squeeze(-1)

    def sample_targets(self, answer_indices: list, n: int, temperature: float = 1.0, uniform_frac: float = 0.2) -> list:
        n_uniform = max(1, int(n * uniform_frac))
        n_adv = n - n_uniform
        uniform_picks = np.random.choice(answer_indices, n_uniform).tolist()
        adv_picks = []
        if n_adv > 0:
            idx = torch.tensor(answer_indices)
            with torch.no_grad():
                logits = self.forward(idx) / max(temperature, 1e-3)
            probs = torch.softmax(logits, dim=0).numpy()
            adv_picks = np.random.choice(answer_indices, size=n_adv, p=probs, replace=True).tolist()
        return uniform_picks + adv_picks

    def update(self, optimizer: torch.optim.Optimizer, chosen_indices: list, agent_rewards: list) -> float:
        if not chosen_indices:
            return 0.0
        idx = torch.tensor(chosen_indices, dtype=torch.long)
        rewards = torch.tensor(agent_rewards, dtype=torch.float32)
        scores = self.forward(idx)
        loss = (scores * rewards).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def difficulty_ranking(self, answer_indices: list, words: list, top_k: int = 20) -> list:
        self.eval()
        with torch.no_grad():
            idx = torch.tensor(answer_indices)
            scores = self.forward(idx).numpy()
        ranked = sorted(zip(answer_indices, scores), key=lambda x: x[1], reverse=True)
        return [(words[i], round(float(s), 4)) for i, s in ranked[:top_k]]
