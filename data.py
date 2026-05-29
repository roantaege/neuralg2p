import random
import torch
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    """Maps tokens to indices and back. Special tokens: pad=0, <s>=1, </s>=2."""
    def __init__(self, tokens):
        special = ["<pad>", "<s>", "</s>"]
        vocab   = special + sorted(set(tokens))
        self.stoi    = {t: i for i, t in enumerate(vocab)}
        self.itos    = {i: t for t, i in self.stoi.items()}
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<s>"]
        self.eos_idx = self.stoi["</s>"]

    def __len__(self):
        return len(self.stoi)


def encode(seq, stoi):
    """Wrap a token sequence with <s> / </s> and convert to indices."""
    return [stoi["<s>"]] + [stoi[t] for t in seq if t in stoi] + [stoi["</s>"]]


def load_cmudict(filepath, seed=42):
    """Load CMUDict and return (train, val, test) as lists of (word, phonemes)."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower().split("(")[0]   # strip variant markers e.g. "(2)"
            data.append((word, parts[1:]))

    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    return data[:int(0.8 * n)], data[int(0.8 * n):int(0.9 * n)], data[int(0.9 * n):]


def collate_fn(pad_idx):
    """Return a collate function that pads src and tgt to the same length."""
    def _collate(batch):
        src, tgt = zip(*batch)
        src = pad_sequence([torch.tensor(x, dtype=torch.long) for x in src],
                           batch_first=True, padding_value=pad_idx)
        tgt = pad_sequence([torch.tensor(x, dtype=torch.long) for x in tgt],
                           batch_first=True, padding_value=pad_idx)
        return src, tgt
    return _collate
