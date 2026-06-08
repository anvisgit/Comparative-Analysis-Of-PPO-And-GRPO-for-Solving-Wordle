import random
import os
# Minimal fallback — only used if wordle_answers.txt is absent.
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
        print("OVERFIT LOL" )
        answers = list(_FALLBACK_ANSWERS)

    answer_set = set(answers)
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
        extra = [w for w in _FALLBACK_EXTRA if w not in answer_set]

    words = answers + extra  

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
