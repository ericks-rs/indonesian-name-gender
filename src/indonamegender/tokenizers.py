"""Char + Word tokenizer (pickled di package data/).

Kunci vocab word-level disimpan sebagai hash, bukan token asli. Vocab dibangun
dari registri admisi dan 23,461 dari 24,945 tokennya tidak muncul di sumber
publik mana pun, sebagian cuma dipakai dua orang. Menyimpan token polos berarti
mengirim potongan korpus itu ke publik. Hash menghilangkan daftarnya tanpa
menyentuh indeks, jadi baris embedding di checkpoint tetap cocok dan prediksinya
identik. Salt ikut dirilis karena paket ini harus bisa jalan, sehingga seseorang
masih bisa menguji satu nama yang sudah dia tebak duluan. Yang hilang adalah
kemampuan menarik seluruh daftarnya sekaligus.
"""
import hashlib
from collections import Counter

VOCAB_SALT = "indonamegender-v1"


def vocab_key(word):
    """Kunci tersimpan untuk satu token word-level."""
    if word.startswith("<") and word.endswith(">"):
        return word
    return hashlib.blake2s((VOCAB_SALT + word).encode("utf-8"),
                           digest_size=8).hexdigest()


class CharTokenizer:
    """Karakter tokenizer dengan PAD + UNK."""
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
    """Word tokenizer dengan min_freq cutoff."""
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.min_freq = min_freq
        self.hashed = True

    def _key(self, word):
        # Pickle lama menyimpan token polos. Dibaca dengan kode ini dia tidak
        # punya atribut hashed, dan lookup-nya harus tetap pakai token polos,
        # kalau tidak semuanya jatuh ke UNK tanpa ada yang error.
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
