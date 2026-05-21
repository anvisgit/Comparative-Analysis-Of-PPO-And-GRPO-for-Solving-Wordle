"""
WordleEnv — improved environment for PPO / GRPO training.

Key upgrades vs original:
  - MAX_GUESSES = 10  (relaxed hard cap; efficiency is incentivised via reward)
  - Shaped reward: +0.15/green tile, +0.05/yellow tile, -0.02/step, +1.0 win, -0.5 fail
  - get_valid_mask()  — boolean array over vocab; impossible words are masked to -inf
  - Richer state (324-dim):
      26 letters × 12 features  (grey | confirmed | green_pos[5] | excl_pos[5])
      + guess-count one-hot (11)
      + remaining-answer fraction (1)
  - compute_feedback() handles duplicate letters correctly
  - is_consistent() re-simulates feedback to filter impossible candidates exactly
"""

import random
import numpy as np

# ── Tile colours ──────────────────────────────────────────────────────────────
GREY   = 0
YELLOW = 1
GREEN  = 2

WORD_LEN    = 5
MAX_GUESSES = 10          # relaxed from 6; env still terminates here if unsolved

# State layout
#   [0 .. 311]  26 × 12 letter-knowledge features
#   [312..322]  guess-count one-hot  (0 … 10)
#   [323]       fraction of answer words still consistent
STATE_DIM = 26 * 12 + (MAX_GUESSES + 1) + 1   # 324


# ── Core feedback logic ───────────────────────────────────────────────────────

def compute_feedback(guess: str, target: str):
    """Return length-5 list of GREY/YELLOW/GREEN; handles duplicate letters."""
    feedback    = [GREY] * WORD_LEN
    target_rem  = list(target)   # letters still available for yellow matching
    guess_rem   = list(guess)

    # First pass: exact matches (green)
    for i in range(WORD_LEN):
        if guess_rem[i] == target_rem[i]:
            feedback[i]   = GREEN
            target_rem[i] = None
            guess_rem[i]  = None

    # Second pass: present but wrong position (yellow)
    for i in range(WORD_LEN):
        if guess_rem[i] is not None and guess_rem[i] in target_rem:
            feedback[i] = YELLOW
            target_rem[target_rem.index(guess_rem[i])] = None

    return feedback


def is_consistent(word: str, guesses: list, feedbacks: list) -> bool:
    """
    Return True iff `word` as the hidden target is consistent with every
    (guess, feedback) pair seen so far.  Uses simulate-and-compare so
    duplicate-letter edge cases are handled exactly.
    """
    for guess, feedback in zip(guesses, feedbacks):
        if compute_feedback(guess, word) != feedback:
            return False
    return True


# ── Environment ───────────────────────────────────────────────────────────────

class WordleEnv:
    """
    Gymnasium-style Wordle environment (no gym dependency).

    Parameters
    ----------
    words   : full guess vocabulary  (list of 5-letter strings)
    answers : subset that can be the hidden target
    """

    def __init__(self, words: list, answers: list):
        self.words       = words
        self.answer_list = answers
        self.action_dim  = len(words)
        self.state_dim   = STATE_DIM

        # fast index lookup  (public for use in training scripts)
        self.word_to_idx    = {w: i for i, w in enumerate(words)}
        self._answer_idx    = [self.word_to_idx[w] for w in answers
                               if w in self.word_to_idx]

        # episode state
        self.target          : str        = ""
        self.guesses_made    : list       = []
        self.feedback_history: list       = []
        self._mask_cache     : np.ndarray = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, target: str = None) -> np.ndarray:
        self.target           = target if target else random.choice(self.answer_list)
        self.guesses_made     = []
        self.feedback_history = []
        self._mask_cache      = None
        return self._build_state()

    def step(self, action: int):
        """
        Parameters
        ----------
        action : index into self.words

        Returns
        -------
        obs, reward, done, info
        """
        word     = self.words[action]
        feedback = compute_feedback(word, self.target)

        # tally known tiles *before* appending this guess
        prev_greens  = sum(f == GREEN  for fb in self.feedback_history for f in fb)
        prev_yellows = sum(f == YELLOW for fb in self.feedback_history for f in fb)

        self.guesses_made.append(word)
        self.feedback_history.append(feedback)
        self._mask_cache = None   # invalidate mask

        won  = all(f == GREEN for f in feedback)
        done = won or (len(self.guesses_made) >= MAX_GUESSES)

        # ── Shaped reward ──────────────────────────────────────────────────
        curr_greens  = sum(f == GREEN  for fb in self.feedback_history for f in fb)
        curr_yellows = sum(f == YELLOW for fb in self.feedback_history for f in fb)

        new_greens  = max(0, curr_greens  - prev_greens)
        new_yellows = max(0, curr_yellows - prev_yellows)

        reward  = new_greens * 0.15 + new_yellows * 0.05
        reward -= 0.02              # step cost → incentivise efficiency
        if won:
            reward += 1.0
        elif done:
            reward -= 0.5

        obs  = self._build_state()
        info = {"won": won, "n_guesses": len(self.guesses_made)}
        return obs, reward, done, info

    def get_valid_mask(self) -> np.ndarray:
        """
        Boolean array of shape (action_dim,).
        False  → word is provably inconsistent with feedback so far.
        Always called AFTER reset / step so the cache is fresh.
        """
        if self._mask_cache is not None:
            return self._mask_cache

        if not self.guesses_made:
            mask = np.ones(self.action_dim, dtype=bool)
        else:
            mask = np.array([
                is_consistent(w, self.guesses_made, self.feedback_history)
                for w in self.words
            ], dtype=bool)

        self._mask_cache = mask
        return mask

    # ── State construction ────────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """
        Returns float32 array of length STATE_DIM (324).

        Letter block (312 = 26 × 12):
            For each letter a-z:
              [0]    grey flag  — confirmed absent (or count-limited)
              [1]    confirmed  — seen as GREEN or YELLOW at least once
              [2-6]  green_pos  — position i is confirmed GREEN for this letter
              [7-11] excl_pos   — position i confirmed NOT this letter (YELLOW seen there)
        """
        letter_feat = np.zeros((26, 12), dtype=np.float32)

        for guess, feedback in zip(self.guesses_made, self.feedback_history):
            for pos, (ch, fb) in enumerate(zip(guess, feedback)):
                li = ord(ch) - ord('a')
                if fb == GREEN:
                    letter_feat[li, 1]       = 1.0   # confirmed in word
                    letter_feat[li, 2 + pos] = 1.0   # green at this position
                elif fb == YELLOW:
                    letter_feat[li, 1]       = 1.0   # confirmed in word
                    letter_feat[li, 7 + pos] = 1.0   # NOT at this position
                else:  # GREY
                    letter_feat[li, 0]       = 1.0   # absent (or excess count)

        # Guess-count one-hot  (indices 0-10)
        n        = len(self.guesses_made)
        guess_oh = np.zeros(MAX_GUESSES + 1, dtype=np.float32)
        guess_oh[min(n, MAX_GUESSES)] = 1.0

        # Fraction of answer words still consistent with feedback
        rem_frac = self._remaining_fraction()

        return np.concatenate([letter_feat.flatten(), guess_oh, [rem_frac]])

    def _remaining_fraction(self) -> float:
        if not self.guesses_made:
            return 1.0
        mask = self.get_valid_mask()
        remaining = sum(1 for i in self._answer_idx if mask[i])
        return remaining / max(1, len(self.answer_list))