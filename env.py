import random
import numpy as np

GREY, YELLOW, GREEN = 0, 1, 2
WORD_LEN = 5
MAX_GUESSES = 6
STATE_DIM = 26 * 12 + (MAX_GUESSES + 1) + 1


def compute_feedback(guess: str, target: str) -> list:
    feedback = [GREY] * WORD_LEN
    target_rem = list(target)
    guess_rem = list(guess)
    for i in range(WORD_LEN):
        if guess_rem[i] == target_rem[i]:
            feedback[i] = GREEN
            target_rem[i] = None
            guess_rem[i] = None
    for i in range(WORD_LEN):
        if guess_rem[i] is not None and guess_rem[i] in target_rem:
            feedback[i] = YELLOW
            target_rem[target_rem.index(guess_rem[i])] = None
    return feedback


def is_consistent(word: str, guesses: list, feedbacks: list) -> bool:
    for guess, feedback in zip(guesses, feedbacks):
        if compute_feedback(guess, word) != feedback:
            return False
    return True


def _encode_feedback(fb: list) -> int:
    return fb[0] * 81 + fb[1] * 27 + fb[2] * 9 + fb[3] * 3 + fb[4]


class WordleEnv:
    def __init__(self, words: list, answers: list, test_answers: list = None, cfg=None):
        self.words = words
        self.answer_list = answers
        self.test_answers = test_answers or []
        self.action_dim = len(words)
        self.state_dim = STATE_DIM
        self.word_to_idx = {w: i for i, w in enumerate(words)}
        self._answer_idx = [self.word_to_idx[w] for w in answers if w in self.word_to_idx]

        if cfg is not None:
            self.include_rem = cfg.include_rem
            self.win_reward = cfg.win_reward
            self.step_cost = cfg.step_cost
            self.green_reward = cfg.green_reward
            self.yellow_reward = cfg.yellow_reward
            self.loss_penalty = cfg.loss_penalty
        else:
            self.include_rem = True
            self.win_reward = 2.0
            self.step_cost = 0.02
            self.green_reward = 0.15
            self.yellow_reward = 0.05
            self.loss_penalty = 0.5

        self.target: str = ""
        self.guesses_made: list = []
        self.feedback_history: list = []
        self._fb_codes: list = []
        self._mask_cache: np.ndarray = None

    def reset(self, target: str = None, test: bool = False) -> np.ndarray:
        if target:
            self.target = target
        elif test and self.test_answers:
            self.target = random.choice(self.test_answers)
        else:
            self.target = random.choice(self.answer_list)
        self.guesses_made = []
        self.feedback_history = []
        self._fb_codes = []
        self._mask_cache = None
        return self._build_state()

    def step(self, action: int):
        word = self.words[action]
        feedback = compute_feedback(word, self.target)
        prev_greens = sum(f == GREEN for fb in self.feedback_history for f in fb)
        prev_yellows = sum(f == YELLOW for fb in self.feedback_history for f in fb)
        self.guesses_made.append(word)
        self.feedback_history.append(feedback)
        self._fb_codes.append(_encode_feedback(feedback))
        self._mask_cache = None
        won = all(f == GREEN for f in feedback)
        done = won or len(self.guesses_made) >= MAX_GUESSES
        curr_greens = sum(f == GREEN for fb in self.feedback_history for f in fb)
        curr_yellows = sum(f == YELLOW for fb in self.feedback_history for f in fb)
        reward = (max(0, curr_greens - prev_greens) * self.green_reward
                  + max(0, curr_yellows - prev_yellows) * self.yellow_reward
                  - self.step_cost)
        if won:
            reward += self.win_reward
        elif done:
            reward -= self.loss_penalty
        return self._build_state(), reward, done, {"won": won, "n_guesses": len(self.guesses_made)}

    def get_valid_mask(self) -> np.ndarray:
        if self._mask_cache is not None:
            return self._mask_cache
        if not self.guesses_made:
            self._mask_cache = np.ones(self.action_dim, dtype=bool)
            return self._mask_cache
        self._mask_cache = np.array(
            [is_consistent(w, self.guesses_made, self.feedback_history) for w in self.words],
            dtype=bool,
        )
        return self._mask_cache

    def _build_state(self) -> np.ndarray:
        letter_feat = np.zeros((26, 12), dtype=np.float32)
        for guess, feedback in zip(self.guesses_made, self.feedback_history):
            for pos, (ch, fb) in enumerate(zip(guess, feedback)):
                li = ord(ch) - ord("a")
                if fb == GREEN:
                    letter_feat[li, 1] = 1.0
                    letter_feat[li, 2 + pos] = 1.0
                elif fb == YELLOW:
                    letter_feat[li, 1] = 1.0
                    letter_feat[li, 7 + pos] = 1.0
                else:
                    letter_feat[li, 0] = 1.0
        n = len(self.guesses_made)
        guess_oh = np.zeros(MAX_GUESSES + 1, dtype=np.float32)
        guess_oh[min(n, MAX_GUESSES)] = 1.0
        rem_frac = self._remaining_fraction() if self.include_rem else 0.0
        return np.concatenate([letter_feat.flatten(), guess_oh, [rem_frac]])

    def _remaining_fraction(self) -> float:
        if not self.guesses_made:
            return 1.0
        mask = self.get_valid_mask()
        remaining = sum(1 for i in self._answer_idx if mask[i])
        return remaining / max(1, len(self.answer_list))
