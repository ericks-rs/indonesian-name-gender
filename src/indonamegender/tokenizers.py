import hashlib
from collections import Counter

VOCAB_SALT = "indonamegender-v1"

def vocab_key(word):
    if word.startswith("<") and word.endswith(">"):
        return word
    return hashlib.blake2s((VOCAB_SALT + word).encode("utf-8"),
                           digest_size=8).hexdigest()

class CharTokenizer:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, names):
        chars = set()
        for name in names:
            chars.update(name.lower())
        for i, ch in enumerate(sorted(chars), start=2):
            self.char2idx[ch] = i
            self.idx2char[i] = ch
        return self

    def encode(self, name, max_len):
        ids = [self.char2idx.get(ch, 1) for ch in name.lower()]
        return ids[:max_len] if len(ids) >= max_len else ids + [0] * (max_len - len(ids))

    @property
    def vocab_size(self):
        return len(self.char2idx)

class WordTokenizer:
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.min_freq = min_freq
        self.hashed = True

    def _key(self, word):

        return vocab_key(word) if getattr(self, "hashed", False) else word

    def fit(self, names):
        counter = Counter()
        for name in names:
            counter.update(name.lower().split())
        for i, (word, freq) in enumerate(
            sorted(counter.items(), key=lambda x: -x[1]), start=2
        ):
            if freq >= self.min_freq:
                self.word2idx[self._key(word)] = i
        return self

    def encode(self, name, max_len):
        ids = [self.word2idx.get(self._key(w), 1) for w in name.lower().split()]
        return ids[:max_len] if len(ids) >= max_len else ids + [0] * (max_len - len(ids))

    @property
    def vocab_size(self):
        return len(self.word2idx)
