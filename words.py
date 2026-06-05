import os
import random
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def _load_file(path: str) -> list:
    words = []
    with open(path, 'r') as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == 5 and all(('a' <= c <= 'z' for c in w)):
                words.append(w)
    return words

def get_default_words(answers_path: str=None, guesses_path: str=None) -> tuple:
    if answers_path is None:
        answers_path = os.path.join(DATA_DIR, 'wordle_answers.txt')
    if guesses_path is None:
        guesses_path = os.path.join(DATA_DIR, 'wordle_guesses_extra.txt')
    seen = set()
    answers = []
    for w in _load_file(answers_path):
        if w not in seen:
            seen.add(w)
            answers.append(w)
    words = list(answers)
    if os.path.exists(guesses_path):
        for w in _load_file(guesses_path):
            if w not in seen:
                seen.add(w)
                words.append(w)
    print(f'[words.py] Unique answers: {len(answers)} | Total vocab: {len(words)}')
    return (words, answers)

def get_split_words(test_fraction: float=0.1, seed: int=42, answers_path: str=None, guesses_path: str=None) -> tuple:
    words, answers = get_default_words(answers_path, guesses_path)
    rng = random.Random(seed)
    shuffled = list(answers)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_answers = shuffled[:n_test]
    train_answers = shuffled[n_test:]
    print(f'[words.py] Split — train: {len(train_answers)} | test: {len(test_answers)} | vocab: {len(words)}')
    return (words, train_answers, test_answers)