# Wordle RL

PPO and GRPO agents for Wordle, compared on information-seeking behaviour.

## Setup

```bash
pip install -r requirements.txt
```

### Word list (required)

Download the real NYT answer list and save it next to the scripts:

```bash
# 2,309 answer words
curl -o wordle_answers.txt https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt

# Optional: 12k valid-guess vocabulary (widens action space)
curl -o wordle_guesses.txt https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt
```

Without `wordle_answers.txt` the code falls back to a 50-word toy list that causes severe overfitting — results will be meaningless.

## Run

```bash
python ppo.py  --run_name default
python grpo.py --run_name default
python compare.py
```

## Multi-seed

```bash
python run_seeds.py --algo both --n_seeds 5
```

Each seed independently randomises the train/test split and model initialisation.

## Fixes applied

| File | Fix |
|---|---|
| `adversary.py` | Sign error corrected — adversary now maximises agent difficulty (was minimising it) |
| `env.py` | Yellow/green reward double-counting fixed — only new unique (letter, position) info is rewarded |
| `grpo.py` | Update loop corrected — removed PPO-style ratio clipping; uses plain policy gradient + KL penalty |
| `grpo.py` | `weight_decay=1e-4` added to Adam optimizer |
| `ppo.py` | `weight_decay=1e-4` added to Adam optimizer |
| `config.py` | `test_fraction` default raised 0.10 → 0.15; `weight_decay` field added |
| `words.py` | Replaced 50-word toy list with real NYT word list loader (falls back gracefully with warning) |

## Expected results (with real word list, 300k episodes)

| Metric | PPO | GRPO |
|---|---|---|
| Test win rate | 72–82% | 68–78% |
| Avg guesses (test) | 3.8–4.2 | 4.0–4.5 |
| Train/test gap | 8–12pp | 10–15pp |
| G1 entropy | 4–6 nats | 5–7 nats |
| G1 CSR | 0.55–0.70 | 0.60–0.75 |

The train/test gap is the main overfitting diagnostic. If it exceeds ~15pp, increase `entropy_coef` or reduce training.
