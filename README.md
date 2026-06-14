# Comparative Analysis of PPO and GRPO for Solving Wordle
## Abstract

Wordle presents a deceptively simple challenge: infer a hidden five-letter word within six attempts using only partial information returned after each guess. Beneath this lies a sequential decision making problem characterized by information acquisition. This project investigates whether modern reinforcement learning algorithms can learn effective Wordle solving strategies.
comparing Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) on a controlled Wordle environment consisting of 2,309 official answer words. While PPO achieves a held-out test win rate of 89.6%, GRPO achieves only 5.8% despite exhibiting marginally stronger information-seeking behavior on its opening move. Through analysis of constraint reduction, policy entropy, sample efficiency, and generalization, we find that the central distinction between the algorithms is not their capacity to gather information, but their ability to transform information into action.
---
Every game of Wordle begins with ignorance. The agent knows only that a word exists. Each guess is therefore both a hypothesis and a question. The colored feedback serves as a conversation between the agent and the environment—a process through which uncertainty is gradually compressed into certainty.
An intelligent agent must perform two competing tasks:
1. Acquire information about the world.
2. Use that information to achieve a goal.
The tension between these objectives is the classical exploration–exploitation dilemma. Reinforcement learning algorithms differ fundamentally in how they navigate this tension.
This work compares PPO and GRPO within this setting and asks a central question:
> Does superior information gathering necessarily lead to superior problem solving?
The answer, as the experiments reveal, is no, lol.
#  Setup
Metrics include:
* Win rate

* Average guesses

* Constraint reduction per guess

* Policy entropy

* First-guess diversity

* Frequency-tier performance

* Training-test generalization gap



---



# 3. Final Results



| Metric          |   PPO | GRPO |

| --------------- | ----: | ---: |

| Test Win Rate   | 89.6% | 5.8% |

| Average Guesses |  4.63 | 3.85 |

| Common Words    | 88.7% | 4.3% |

| Medium Words    | 93.9% | 6.1% |

| Rare Words      | 86.3% | 6.8% |


At first glance, GRPO appears more efficient because it requires fewer guesses on successful games. However, this interpretation is misleading. A solver that rarely succeeds can achieve a low average guess count simply because most games terminate in failure before all six guesses are used. Therefore, win rate remains the primary measure of performance. Under this criterion, PPO overwhelmingly dominates.
---

# Sample Efficiency

One of the most surprising observations emerges during training.

### PPO



| Episodes | Test Win Rate |

| -------: | ------------: |

|       2k |         84.2% |

|       5k |         87.0% |

|      10k |         85.9% |

|      20k |         93.4% |

|      50k |         90.2% |



PPO exhibits stable learning dynamics and converges toward a robust policy.



### GRPO



| Episodes | Test Win Rate |

| -------: | ------------: |

|       2k |         88.5% |

|       5k |         90.8% |

|      10k |         90.2% |

|      20k |         89.3% |

|      50k |         17.0% |

|     100k |          1.7% |

|     300k |          5.8% |



GRPO initially performs exceptionally well. Then it collapses. This collapse represents the most significant finding of the study. The algorithm demonstrates that learning is not synonymous with improvement. Beyond a certain point, optimization ceases to refine behavior and instead destroys it. More gradient updates do not necessarily produce more intelligence.
Sometimes they produce less.
---
# Constraint Reduction Analysis
Constraint Reduction Score (CSR) measures the fraction of candidate words eliminated after each guess.
| Guess |   PPO |  GRPO |

| ----- | ----: | ----: |

| 1     | 0.917 | 0.946 |

| 2     | 0.702 | 0.726 |

| 3     | 0.360 | 0.245 |

| 4     | 0.149 | 0.040 |

| 5     | 0.054 | 0.009 |

GRPO achieves superior elimination during the first two guesses. This initially appears promising. However, the advantage rapidly disappears.By Guess 3, PPO becomes substantially more effective at reducing uncertainty.The interpretation is subtle.
GRPO learns to ask good questions.
PPO learns to ask useful questions.
---
# Entropy and the Dynamics of Belief
Policy entropy provides a window into the agent's confidence.

### PPO



| Guess | Entropy |

| ----- | ------: |

| 1     |    9.47 |

| 2     |    5.98 |

| 3     |    3.43 |

| 4     |    1.80 |

| 5     |    1.03 |

| 6     |    0.73 |



Entropy decreases steadily.The agent begins uncertain and becomes progressively more certain as evidence accumulates. This is the signature of successful inference.



### GRPO



| Guess | Entropy |

| ----- | ------: |

| 2     |    0.83 |

| 3     |    4.10 |

| 4     |    7.77 |

| 5     |    9.07 |

| 6     |    9.47 |



The trend is reversed.The agent becomes increasingly uncertain over time.Rather than converging toward an answer, its belief distribution expands.The environment reveals more information, yet the policy becomes less decisive.
---
# First-Guess Collapse
### PPO
Opening word:  splay
### GRPO
Opening word:  bream
First-guess entropy equals zero for both policies. Interestingly, this collapse does not significantly harm PPO. Wordle contains a small number of highly informative opening moves, making deterministic openings entirely reasonable. The critical difference lies not in how the game begins, but in how the policy adapts afterward.
PPO remains flexible.
GRPO does not.
---
# Generalization
GRPO exhibits:

* Training Win Rate: 18.4%

* Test Win Rate: 5.8%

* Generalization Gap: 12.7 percentage points

The gap indicates overfitting.

The policy learns patterns specific to the training distribution but fails to transfer them to unseen answers.

PPO's high held-out performance demonstrates the opposite behavior.
---



# 9. Discussion

The original hypothesis proposed that GRPO might develop stronger information-seeking behavior than PPO.
The results partially support this claim.
GRPO achieves a higher first-guess constraint reduction score:

* PPO: 0.917

* GRPO: 0.946
---

# Conclusion

This study demonstrates a clear performance advantage for PPO over GRPO in the Wordle domain.
PPO achieves:
* 89.6% held-out win rate

* Stable training dynamics

* Strong generalization

* Progressive entropy reduction

* Effective late-game reasoning

GRPO demonstrates:

* Slightly stronger early information gathering

* Severe policy collapse during extended training

* Poor generalization

* Increasing uncertainty over time

* Extremely low final success rates









# 2,309 answer words
curl -o wordle_answers.txt https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt

# Optional: 12k valid-guess vocabulary (widens action space)
curl -o wordle_guesses.txt https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt
