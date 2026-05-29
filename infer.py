"""
infer.py  –  G2P inference plugin
Usage (CLI):
    python infer.py hello
    python infer.py --checkpoint best_model.pt psychology

Usage (import):
    from neuralg2p import G2PInference
    g2p = G2PInference("best_model.pt")
    g2p("hello")                              # single word
    g2p.pronounce_batch(["the", "quick"])     # multiple words
"""

import argparse
import torch

try:
    from .model import G2P
except ImportError:
    from model import G2P


class G2PInference:
    def __init__(self, checkpoint_path="best_model.pt"):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        self.src_stoi = ckpt["src_vocab"]
        self.tgt_itos = {i: t for t, i in ckpt["tgt_vocab"].items()}
        config = ckpt["config"]
        config.cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if config.cuda else "cpu")
        self.model = G2P(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def __call__(self, word):
        """Return a list of ARPAbet phonemes for a single word string."""
        indices = (
            [self.src_stoi["<s>"]]
            + [self.src_stoi[c] for c in word.lower() if c in self.src_stoi]
            + [self.src_stoi["</s>"]]
        )
        src = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(1)
        with torch.no_grad():
            hyp = self.model(src)
        return [self.tgt_itos[i] for i in hyp if i in self.tgt_itos]

    def pronounce_batch(self, words):
        """Return a list of phoneme lists for a list of word strings."""
        return [self(w) for w in words]


def main():
    parser = argparse.ArgumentParser(description="G2P phoneme prediction")
    parser.add_argument("word", help="Word to convert to phonemes")
    parser.add_argument("--checkpoint", default="best_model.pt")
    args = parser.parse_args()

    g2p = G2PInference(args.checkpoint)
    phonemes = g2p(args.word)
    print(f"{args.word} -> {' '.join(phonemes)}")


if __name__ == "__main__":
    main()
