import random
import os

# ---------------------------------------------------------------------------
# Real NYT answer list loader
# ---------------------------------------------------------------------------
# Download the answer list (2,309 words) from:
#   https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b
# Save it as  wordle_answers.txt  (one word per line) in the same directory.
#
# Optionally also save a larger guess vocabulary as  wordle_guesses.txt
# (e.g. the 12k valid-guess list from the same author's gists).
# That file is used to widen the action space but is never a training target.
# ---------------------------------------------------------------------------

# Minimal fallback — only used if wordle_answers.txt is absent.
# DO NOT rely on this for real experiments; it will cause severe overfitting.
_FALLBACK_ANSWERS = [
    "about","other","which","their","there","apple","place","right","think","could",
    "would","where","light","large","small","world","never","under","ocean","house",
    "money","water","happy","craft","speed","print","field","green","brown","black",
    "white","heart","laugh","judge","dream","night","early","grain","scene","begin",
    "bring","break","climb","trust","share","guide","train","chain","spice","candy",
]

_FALLBACK_EXTRA = [
    "adieu","arise","stare","slate","crate","slant","trace","pride","brink","flame",
    "stone","grace","shout","bland","crane","shale","whale","bloom","brave","faint",
    "gamer","piano","zesty","quilt","vapid","xenon","fuzzy","jazzy","pixel","crypt",
    "nymph","fjord","glyph","briar","caper","demon","enact",
]


def get_split_words(test_fraction=0.15, split_seed=0, answers_path=None, guesses_path=None):
    """Return (words, train_answers, test_answers).

    words         — full vocabulary (answers + extra guesses); used as the action space.
    train_answers — targets seen during training.
    test_answers  — held-out targets for evaluation (never in training).

    File locations (resolved relative to this module):
        answers_path  defaults to  <dir>/wordle_answers.txt
        guesses_path  defaults to  <dir>/wordle_guesses.txt  (optional)
    """
    base = os.path.dirname(os.path.abspath(__file__))

    if answers_path is None:
        answers_path = os.path.join(base, "wordle_answers.txt")
    if guesses_path is None:
        guesses_path = os.path.join(base, "wordle_guesses.txt")

    # --- load answers ---
    if os.path.exists(answers_path):
        with open(answers_path) as f:
            answers = [
                line.strip().lower() for line in f
                if len(line.strip()) == 5 and line.strip().isalpha()
            ]
        answers = list(dict.fromkeys(answers))  # deduplicate, preserve order
    else:
        print(
            f"[words] WARNING: {answers_path} not found — using fallback 50-word list.\n"
            "        Download the real NYT list from:\n"
            "        https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b\n"
            "        and save it as wordle_answers.txt to avoid severe overfitting."
        )
        answers = list(_FALLBACK_ANSWERS)

    answer_set = set(answers)

    # --- load extra guess vocabulary (optional) ---
    extra: list[str] = []
    if os.path.exists(guesses_path):
        with open(guesses_path) as f:
            extra = [
                line.strip().lower() for line in f
                if len(line.strip()) == 5 and line.strip().isalpha()
                and line.strip().lower() not in answer_set
            ]
        extra = list(dict.fromkeys(extra))
    else:
        # fall back to the hardcoded extra guesses
        extra = [w for w in _FALLBACK_EXTRA if w not in answer_set]

    words = answers + extra  # answers first so indices are stable

    # --- train / test split ---
    rng = random.Random(split_seed)
    shuffled = list(answers)
    rng.shuffle(shuffled)
    n_test = max(50, int(len(shuffled) * test_fraction))  # minimum 50 test words
    test_answers = shuffled[:n_test]
    train_answers = shuffled[n_test:]

    print(
        f"[words] train={len(train_answers)} test={len(test_answers)} "
        f"vocab={len(words)} split_seed={split_seed}"
    )
    return words, train_answers, test_answers
