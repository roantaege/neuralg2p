"""
train.py  –  train the G2P model and save best_model.pt
Usage:
    python train.py
    python train.py --epochs 100 --d_hidden 1024 --no_attention
"""

import argparse
import os
import time
import requests
import zipfile
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import G2P
from data import Vocab, load_cmudict, encode, collate_fn
from metrics import phoneme_error_rate


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train G2P seq2seq model")
    p.add_argument("--data_path",    default="data/cmudict/")
    p.add_argument("--checkpoint",   default="best_model.pt")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=100)
    p.add_argument("--max_len",      type=int,   default=20)
    p.add_argument("--beam_size",    type=int,   default=3)
    p.add_argument("--d_embed",      type=int,   default=500)
    p.add_argument("--d_hidden",     type=int,   default=500)
    p.add_argument("--no_attention", action="store_true")
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--lr",           type=float, default=0.007)
    p.add_argument("--lr_decay",     type=float, default=0.5)
    p.add_argument("--lr_min",       type=float, default=1e-5)
    p.add_argument("--n_bad_loss",   type=int,   default=5)
    p.add_argument("--clip",         type=float, default=2.3)
    p.add_argument("--seed",         type=int,   default=5)
    p.add_argument("--no_cuda",      action="store_true")
    return p.parse_args()


# ── Data download ─────────────────────────────────────────────────────────────

def maybe_download(data_path):
    if os.path.isdir(data_path):
        return
    os.makedirs("data", exist_ok=True)
    url      = "https://github.com/cmusphinx/cmudict/archive/master.zip"
    zip_path = "data/cmudict.zip"
    print("Downloading CMUDict...")
    r = requests.get(url)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/")
    shutil.move("data/cmudict-master", data_path)
    print("Done.")


# ── Training ──────────────────────────────────────────────────────────────────

def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for src, tgt in val_iter:
            if cuda:
                src, tgt = src.cuda(), tgt.cuda()
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            output, _, _ = model(src, tgt[:-1, :])
            output = output.reshape(-1, output.size(-1))
            target = tgt[1:, :].reshape(-1)
            val_loss += criterion(output, target).item() * src.size(1)
    return val_loss / len(val_iter.dataset)


def train(args, train_iter, val_iter, model, criterion, optimizer,
          src_vocab, tgt_vocab):
    state = dict(iteration=0, n_total=0, train_loss=0.0,
                 n_bad_loss=0, best_val_loss=float("inf"),
                 stop=False, init=time.time())

    for epoch in range(1, args.epochs + 1):
        print(f"=> EPOCH {epoch}")
        for src, tgt in train_iter:
            state["iteration"] += 1
            model.train()

            if args.cuda:
                src, tgt = src.cuda(), tgt.cuda()
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            output, _, _ = model(src, tgt[:-1, :])
            output = output.reshape(-1, output.size(-1))
            target = tgt[1:, :].reshape(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            state["n_total"]    += src.size(1)
            state["train_loss"] += loss.item() * src.size(1)

            if state["iteration"] % args.log_every == 0:
                train_loss = state["train_loss"] / state["n_total"]
                val_loss   = validate(val_iter, model, criterion)
                elapsed    = time.time() - state["init"]
                print(f"   % Time: {elapsed:5.0f} | Iter: {state['iteration']:5} "
                      f"| Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                state["n_total"] = state["train_loss"] = 0

                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    state["n_bad_loss"]    = 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "config":      args,
                        "src_vocab":   src_vocab.stoi,
                        "tgt_vocab":   tgt_vocab.stoi,
                    }, args.checkpoint)
                    print(f"   => Saved {args.checkpoint}")
                else:
                    state["n_bad_loss"] += 1

                if state["n_bad_loss"] >= args.n_bad_loss:
                    state["best_val_loss"] = val_loss
                    state["n_bad_loss"]    = 0
                    for pg in optimizer.param_groups:
                        pg["lr"] *= args.lr_decay
                    new_lr = optimizer.param_groups[0]["lr"]
                    print(f"=> LR → {new_lr:.2e}")
                    if new_lr < args.lr_min:
                        state["stop"] = True
                        return

        if state["stop"]:
            print("=> Early stop: LR below minimum.")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    args.cuda      = not args.no_cuda and torch.cuda.is_available()
    args.attention = not args.no_attention

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    maybe_download(args.data_path)

    filepath = os.path.join(args.data_path, "cmudict.dict")
    train_raw, val_raw, _ = load_cmudict(filepath, seed=args.seed)

    # build vocab from training data only
    src_tokens, tgt_tokens = [], []
    for w, p in train_raw:
        src_tokens.extend(list(w))
        tgt_tokens.extend(p)
    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)

    def encode_split(data):
        return [(encode(list(w), src_vocab.stoi), encode(p, tgt_vocab.stoi))
                for w, p in data]

    train_data = encode_split(train_raw)
    val_data   = encode_split(val_raw)

    pad_idx    = src_vocab.pad_idx
    _collate   = collate_fn(pad_idx)
    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True,  collate_fn=_collate)
    val_iter   = DataLoader(val_data,   batch_size=args.batch_size,
                            shuffle=False, collate_fn=_collate)

    args.g_size = len(src_vocab)
    args.p_size = len(tgt_vocab)

    model     = G2P(args)
    criterion = nn.NLLLoss(ignore_index=pad_idx)
    if args.cuda:
        model.cuda()
        criterion.cuda()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    train(args, train_iter, val_iter, model, criterion, optimizer,
          src_vocab, tgt_vocab)

    print("Training complete.")
