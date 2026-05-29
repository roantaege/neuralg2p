import requests
import zipfile
import shutil
import argparse
from torch.utils.data import Dataset
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import random


# ── Data loading ────────────────────────────────────────────────────────────

def load_cmudict(filepath, seed=42):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            # strip trailing variant markers like "(2)"
            word = word.split("(")[0]
            phonemes = parts[1:]
            data.append((word, phonemes))

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train = data[:int(0.8 * n)]
    val   = data[int(0.8 * n):int(0.9 * n)]
    test  = data[int(0.9 * n):]
    return train, val, test


class Vocab:
    def __init__(self, tokens):
        # index 0 = pad, 1 = <s>, 2 = </s>
        special = ["<pad>", "<s>", "</s>"]
        vocab = special + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<s>"]
        self.eos_idx = self.stoi["</s>"]

    def __len__(self):
        return len(self.stoi)


def encode(seq, stoi):
    return (
        [stoi["<s>"]]
        + [stoi[t] for t in seq if t in stoi]
        + [stoi["</s>"]]
    )


# ── Config ───────────────────────────────────────────────────────────────────

parser = {
    'data_path':        '../data/cmudict/',
    'epochs':           50,
    'batch_size':       100,
    'max_len':          20,
    'beam_size':        3,
    'd_embed':          500,
    'd_hidden':         500,
    'attention':        True,
    'log_every':        100,
    'lr':               0.007,
    'lr_decay':         0.5,
    'lr_min':           1e-5,
    'n_bad_loss':       5,
    'clip':             2.3,
    'cuda':             True,
    'seed':             5,
    'intermediate_path': '../intermediate/g2p/',
}
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()

os.makedirs(args.intermediate_path, exist_ok=True)

if not os.path.isdir(args.data_path):
    os.makedirs("../data", exist_ok=True)
    url      = "https://github.com/cmusphinx/cmudict/archive/master.zip"
    zip_path = "../data/cmudict.zip"
    response = requests.get(url)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("../data/")
    shutil.move("../data/cmudict-master", args.data_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# ── Model ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm      = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden  = d_hidden

    def forward(self, x_seq):
        # x_seq: (seq_len, batch)
        e_seq     = self.embedding(x_seq)           # seq x batch x dim
        batch     = x_seq.size(1)
        h = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        c = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        outputs = []
        for e in e_seq.chunk(e_seq.size(0), dim=0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            outputs.append(h)
        return torch.stack(outputs, 0), h, c
    # Based on https://github.com/OpenNMT/OpenNMT-py


class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x, context=None):
        if context is None:
            return x
        # x:       (batch, dim)
        # context: (batch, seq, dim)
        assert x.size(0) == context.size(0)
        assert x.size(1) == context.size(2)
        attn = F.softmax(
            context.bmm(x.unsqueeze(2)).squeeze(2), dim=1
        )                                                       # (batch, seq)
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)  # (batch, dim)
        o = self.linear(torch.cat((x, weighted_context), dim=1))
        return torch.tanh(o)   # FIX: was returning x, discarding the attention output


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm      = nn.LSTMCell(d_embed, d_hidden)
        self.attn      = Attention(d_hidden)
        self.linear    = nn.Linear(d_hidden, vocab_size)

    def forward(self, x_seq, h, c, context=None):
        e_seq = self.embedding(x_seq)
        outputs = []
        for e in e_seq.chunk(e_seq.size(0), dim=0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            outputs.append(self.attn(h, context))
        o = torch.stack(outputs, 0)                         # (seq, batch, dim)
        o = self.linear(o.view(-1, h.size(1)))              # (seq*batch, vocab)
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h, c


class G2P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config.g_size, config.d_embed, config.d_hidden)
        self.decoder = Decoder(config.p_size, config.d_embed, config.d_hidden)
        self.config  = config

    def forward(self, g_seq, p_seq=None):
        o, h, c = self.encoder(g_seq)
        context = o.transpose(0, 1) if self.config.attention else None
        if p_seq is not None:
            return self.decoder(p_seq, h, c, context)
        else:
            assert g_seq.size(1) == 1, "Generation requires batch size of 1"  # FIX: was == 0
            return self._generate(h, c, context)

    def _generate(self, h, c, context):
        beam = Beam(
            self.config.beam_size,
            pad=0,   # must match Vocab indices
            bos=1,
            eos=2,
            cuda=self.config.cuda,
        )
        h       = h.expand(beam.size, h.size(1))
        c       = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))

        for _ in range(self.config.max_len):
            x = beam.get_current_state()
            o, h, c = self.decoder(x.unsqueeze(0), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h = h.index_select(0, beam.get_current_origin())
            c = c.index_select(0, beam.get_current_origin())

        return beam.get_hyp(0)
    # Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/


class Beam:
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=0, bos=1, eos=2, cuda=False):
        self.size  = size
        self.done  = False
        self.pad   = pad
        self.bos   = bos
        self.eos   = eos
        device     = torch.device("cuda" if cuda else "cpu")

        self.scores  = torch.zeros(size, device=device)
        self.prevKs  = []
        self.nextYs  = [torch.full((size,), pad, dtype=torch.long, device=device)]
        self.nextYs[0][0] = bos

    def get_current_state(self):
        return self.nextYs[-1]

    def get_current_origin(self):
        return self.prevKs[-1]

    def advance(self, word_lk):
        num_words = word_lk.size(1)
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk           = beam_lk.view(-1)
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, largest=True, sorted=True)
        self.scores = bestScores

        prev_k = bestScoresId // num_words  # FIX: was / (float div in Python 3)
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k].item())
            k = self.prevKs[j][k]
        return hyp[::-1]
    # Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py


# ── Metrics ──────────────────────────────────────────────────────────────────

import Levenshtein  # pip install python-Levenshtein

def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c     = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1  = [chr(p2c[p]) for p in p_seq1]
    c_seq2  = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance("".join(c_seq1), "".join(c_seq2)) / len(c_seq2)


# ── Training helpers ─────────────────────────────────────────────────────────

def adjust_learning_rate(optimizer, lr_decay):
    for pg in optimizer.param_groups:
        pg["lr"] *= lr_decay


def train(config, train_iter, val_iter, model, criterion, optimizer, epoch,
          state):
    """
    state: mutable dict holding  iteration, n_total, train_loss,
                                  n_bad_loss, best_val_loss, stop
    """
    print(f"=> EPOCH {epoch}")
    for batch in train_iter:
        state["iteration"] += 1
        model.train()

        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        input_tgt = tgt[:-1, :]
        target    = tgt[1:, :]

        output, _, _ = model(src, input_tgt)
        output = output.reshape(-1, output.size(-1))
        target = target.reshape(-1)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # FIX: use _() form
        optimizer.step()

        batch_size = src.size(1)
        state["n_total"]    += batch_size
        state["train_loss"] += loss.item() * batch_size

        if state["iteration"] % config.log_every == 0:
            train_loss = state["train_loss"] / state["n_total"]
            val_loss   = validate(val_iter, model, criterion)
            print(
                f"   % Time: {time.time()-state['init']:5.0f} | "
                f"Iter: {state['iteration']:5} | "
                f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
            )
            state["n_total"] = state["train_loss"] = 0

            if val_loss < state["best_val_loss"]:
                state["best_val_loss"] = val_loss
                state["n_bad_loss"]    = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config":      config,
                        "src_vocab":   src_vocab.stoi,
                        "tgt_vocab":   tgt_vocab.stoi,
                    },
                    "best_model.pt",
                )
                print("   => Saved best model")
            else:
                state["n_bad_loss"] += 1  # FIX: was inside the if block (no-op)

            if state["n_bad_loss"] >= config.n_bad_loss:
                state["best_val_loss"] = val_loss
                state["n_bad_loss"]    = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]["lr"]
                print(f"=> Adjusted LR to: {new_lr}")
                if new_lr < config.lr_min:
                    state["stop"] = True
                    return


def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_iter:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            input_tgt = tgt[:-1, :]
            target    = tgt[1:, :]
            output, _, _ = model(src, input_tgt)
            output = output.reshape(-1, output.size(-1))
            target = target.reshape(-1)
            loss   = criterion(output, target)
            val_loss += loss.item() * src.size(1)
    return val_loss / len(val_iter.dataset)


def test(model, test_iter, tgt_vocab, criterion):
    model.eval()
    total_per = total_wer = n = 0
    with torch.no_grad():
        for src, tgt in test_iter:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            input_tgt = tgt[:-1, :]
            target    = tgt[1:, :]
            output, _, _ = model(src, input_tgt)
            output = output.argmax(-1)
            for i in range(output.size(1)):
                pred = output[:, i].tolist()
                gold = target[:, i].tolist()
                total_per += phoneme_error_rate(pred, gold)
                total_wer += int(pred != gold)
                n += 1
    print(f"Phoneme error rate (PER): {total_per/n*100:.2f}%")
    print(f"Word error rate    (WER): {total_wer/n*100:.2f}%")


# ── Dataset / vocab / loaders ─────────────────────────────────────────────────

filepath = os.path.join(args.data_path, "cmudict.dict")
train_data, val_data, test_data = load_cmudict(filepath, seed=args.seed)

src_tokens, tgt_tokens = [], []
for w, p in train_data:
    src_tokens.extend(list(w))
    tgt_tokens.extend(p)

src_vocab = Vocab(src_tokens)
tgt_vocab = Vocab(tgt_tokens)


def encode_dataset(data, sv, tv):
    return [(encode(list(w), sv.stoi), encode(p, tv.stoi)) for w, p in data]


train_data = encode_dataset(train_data, src_vocab, tgt_vocab)
val_data   = encode_dataset(val_data,   src_vocab, tgt_vocab)
test_data  = encode_dataset(test_data,  src_vocab, tgt_vocab)


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence([torch.tensor(x, dtype=torch.long) for x in src],
                       batch_first=True, padding_value=src_vocab.pad_idx)
    tgt = pad_sequence([torch.tensor(x, dtype=torch.long) for x in tgt],
                       batch_first=True, padding_value=tgt_vocab.pad_idx)
    return src, tgt


train_iter = DataLoader(train_data, batch_size=args.batch_size,
                        shuffle=True,  collate_fn=collate_fn)
val_iter   = DataLoader(val_data,   batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)
test_iter  = DataLoader(test_data,  batch_size=1,
                        shuffle=False, collate_fn=collate_fn)

# ── Build model ───────────────────────────────────────────────────────────────

config         = args
config.g_size  = len(src_vocab)
config.p_size  = len(tgt_vocab)

model     = G2P(config)
criterion = nn.NLLLoss(ignore_index=src_vocab.pad_idx)
if config.cuda:
    model.cuda()
    criterion.cuda()
optimizer = optim.Adagrad(model.parameters(), lr=config.lr)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Evaluate a previously saved checkpoint
        checkpoint = torch.load("best_model.pt")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        test(model, test_iter, tgt_vocab, criterion)
    else:
        # Train from scratch
        state = dict(
            iteration=0, n_total=0, train_loss=0.0,
            n_bad_loss=0, best_val_loss=float("inf"),
            stop=False, init=time.time(),
        )
        for epoch in range(1, config.epochs + 1):
            train(config, train_iter, val_iter, model, criterion,
                  optimizer, epoch, state)
            if state["stop"]:
                print("=> Early stop: LR below minimum.")
                break

        print("Training complete. Running test evaluation...")
        checkpoint = torch.load("best_model.pt")
        model.load_state_dict(checkpoint["model_state"])
        test(model, test_iter, tgt_vocab, criterion)