import argparse
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    algo: str = "grpo"
    run_name: str = "default"
    seed: int = 42
    split_seed: int = 0
    hidden: int = 256
    use_attention: bool = False
    reward_type: str = "shaped"
    win_reward: float = 2.0
    step_cost: float = 0.02
    green_reward: float = 0.15
    yellow_reward: float = 0.05
    loss_penalty: float = 0.5
    max_guesses: int = 6
    test_fraction: float = 0.15
    total_episodes: int = 300_000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.05
    n_epochs: int = 4
    batch_size: int = 128
    group_size: int = 16
    eps_per_update: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    vf_coef: float = 0.5
    rollout_steps: int = 1024
    total_steps: int = 300_000
    ckpt_dir: str = "checkpoints"
    log_interval: int = 10
    show_game_every: int = 500
    device: str = "cpu"
    n_seeds: int = 1

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def ckpt_path(self, suffix=""):
        return os.path.join(self.ckpt_dir, f"{self.algo}_{self.run_name}{suffix}")


def parse_args(algo="grpo"):
    cfg = ExperimentConfig(algo=algo)
    p = argparse.ArgumentParser(f"{algo.upper()} Wordle Trainer")
    for fname, fval in asdict(cfg).items():
        ftype = type(fval)
        if ftype == bool:
            p.add_argument(f"--{fname}", type=lambda x: x.lower() != "false", default=fval)
        else:
            p.add_argument(f"--{fname}", type=ftype, default=fval)
    return ExperimentConfig(**vars(p.parse_args()))
