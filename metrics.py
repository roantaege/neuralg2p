import Levenshtein  # pip install python-Levenshtein


def phoneme_error_rate(p_seq1, p_seq2):
    """Levenshtein-based phoneme error rate between two phoneme sequences."""
    p_vocab = set(p_seq1 + p_seq2)
    p2c     = {p: i for i, p in enumerate(p_vocab)}
    c1 = "".join(chr(p2c[p]) for p in p_seq1)
    c2 = "".join(chr(p2c[p]) for p in p_seq2)
    return Levenshtein.distance(c1, c2) / len(c2)
    # Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
