# Wordle RL

Compact Wordle RL with PPO and GRPO training.

## Run

pip install -r requirements.txt
python ppo.py --run_name default
python grpo.py --run_name default
python compare.py

## Multi-seed

python run_seeds.py --algo both --n_seeds 5
