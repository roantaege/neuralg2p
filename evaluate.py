"""
evaluate.py  –  load a saved checkpoint, report PER/WER, show examples
Usage:
    python evaluate.py
    python evaluate.py --checkpoint best_model.pt --n_examples 10 --beam_size 1
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g2p import G2P, load_cmudict, encode, collate_fn, phoneme_error_rate


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="best_model.pt")
    p.add_argument("--data_path",    default="data/cmudict/")
    p.add_argument("--seed",         type=int, default=5)
    p.add_argument("--n_examples",   type=int, default=10)
    p.add_argument("--batch_size",   type=int, default=128,
                   help="Encoder batch size (beam search still runs per-example)")
    p.add_argument("--beam_size",    type=int, default=None,
                   help="Override beam size (1 = greedy, fastest)")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Cap test set size for quick runs")
    return p.parse_args()


def run_test(model, test_data_raw, src_stoi, tgt_itos, config, batch_size=128):
    """Batched encoder + per-example beam search decoder."""
    model.eval()
    total_per = total_wer = n = 0
    pad_idx = src_stoi["<pad>"]

    with torch.no_grad():
        for i in range(0, len(test_data_raw), batch_size):
            chunk = test_data_raw[i:i + batch_size]
            words, phonemes_batch = zip(*chunk)

            # pad and batch the encoder inputs
            import torch
            from torch.nn.utils.rnn import pad_sequence
            encoded = [torch.tensor(encode(list(w), src_stoi), dtype=torch.long)
                       for w in words]
            src = pad_sequence(encoded, batch_first=False, padding_value=pad_idx)
            if config.cuda:
                src = src.cuda()

            enc_out, h, c = model.encoder(src)  # enc_out: (seq, batch, hidden)

            for j in range(len(chunk)):
                hj    = h[j].unsqueeze(0)
                cj    = c[j].unsqueeze(0)
                ctx_j = enc_out[:, j, :].unsqueeze(0)   # (1, seq, hidden)

                pred_ids = model._generate(hj, cj, ctx_j)

                pred = [tgt_itos[idx] for idx in pred_ids
                        if tgt_itos.get(idx, "") not in ("<s>", "</s>", "<pad>")]
                gold = list(phonemes_batch[j])

                total_per += phoneme_error_rate(pred, gold)
                total_wer += int(pred != gold)
                n += 1

    print(f"Phoneme error rate (PER): {total_per/n*100:.2f}")
    print(f"Word error rate    (WER): {total_wer/n*100:.2f}")


def show(word, phonemes, model, src_stoi, tgt_itos, config):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(encode(list(word), src_stoi),
                           dtype=torch.long).unsqueeze(1)
        if config.cuda:
            src = src.cuda()
        pred_ids = model(src)

    pred = [tgt_itos[i] for i in pred_ids
            if tgt_itos.get(i, "") not in ("<s>", "</s>", "<pad>")]
    print(f"> {word}")
    print(f"= {' '.join(phonemes)}")
    print(f"< {' '.join(pred)}")
    print()


if __name__ == "__main__":
    args = get_args()

    checkpoint  = torch.load(args.checkpoint, map_location="cpu")
    config      = checkpoint["config"]
    config.cuda = getattr(config, "cuda", False) and torch.cuda.is_available()
    if args.beam_size is not None:
        config.beam_size = args.beam_size

    src_stoi = checkpoint["src_vocab"]
    tgt_stoi = checkpoint["tgt_vocab"]
    tgt_itos = {i: t for t, i in tgt_stoi.items()}
    config.g_size = len(src_stoi)
    config.p_size = len(tgt_stoi)

    model = G2P(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if config.cuda:
        model.cuda()

    filepath = os.path.join(args.data_path, "cmudict.dict")
    _, _, test_raw = load_cmudict(filepath, seed=args.seed)
    if args.max_examples:
        test_raw = test_raw[:args.max_examples]

    print(f"Evaluating {len(test_raw)} examples "
          f"(beam_size={config.beam_size}, batch_size={args.batch_size})...")
    run_test(model, test_raw, src_stoi, tgt_itos, config,
             batch_size=args.batch_size)

    print(f"\nShowing {args.n_examples} examples  (> word  = true  < predicted)\n")
    for i, (word, phonemes) in enumerate(test_raw):
        show(word, phonemes, model, src_stoi, tgt_itos, config)
        if i + 1 == args.n_examples:
            break
