

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import Levenshtein  # pip install python-Levenshtein


torch.serialization.add_safe_globals([argparse.Namespace])

# ── Data ──────────────────────────────────────────────────────────────────────

def load_cmudict(filepath, seed=42):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower().split("(")[0]
            data.append((word, parts[1:]))
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    return data[int(0.9 * n):]  # test split only


def encode(seq, stoi):
    return [stoi["<s>"]] + [stoi[t] for t in seq if t in stoi] + [stoi["</s>"]]


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm      = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden  = d_hidden

    def forward(self, x_seq):
        e_seq = self.embedding(x_seq)
        batch = x_seq.size(1)
        h = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        c = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        outputs = []
        for e in e_seq.chunk(e_seq.size(0), dim=0):
            h, c = self.lstm(e.squeeze(0), (h, c))
            outputs.append(h)
        return torch.stack(outputs, 0), h, c


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x, context=None):
        if context is None:
            return x
        attn = F.softmax(context.bmm(x.unsqueeze(2)).squeeze(2), dim=1)
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)
        return torch.tanh(self.linear(torch.cat((x, weighted_context), dim=1)))


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
            h, c = self.lstm(e.squeeze(0), (h, c))
            outputs.append(self.attn(h, context))
        o = self.linear(torch.stack(outputs, 0).view(-1, h.size(1)))
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h, c


class Beam:
    def __init__(self, size, pad=0, bos=1, eos=2, cuda=False):
        self.size   = size
        self.done   = False
        self.eos    = eos
        device      = torch.device("cuda" if cuda else "cpu")
        self.scores = torch.zeros(size, device=device)
        self.prevKs = []
        self.nextYs = [torch.full((size,), pad, dtype=torch.long, device=device)]
        self.nextYs[0][0] = bos

    def get_current_state(self):  return self.nextYs[-1]
    def get_current_origin(self): return self.prevKs[-1]

    def advance(self, word_lk):
        num_words = word_lk.size(1)
        beam_lk   = (word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
                     if self.prevKs else word_lk[0])
        bestScores, bestScoresId = beam_lk.view(-1).topk(self.size, largest=True, sorted=True)
        self.scores = bestScores
        prev_k      = bestScoresId // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        self.done = (self.nextYs[-1][0] == self.eos)
        return self.done

    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k].item())
            k = self.prevKs[j][k]
        return hyp[::-1]


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
        assert g_seq.size(1) == 1, "Generation requires batch size of 1"
        return self._generate(h, c, context)

    def _generate(self, h, c, context):
        cfg  = self.config
        beam = Beam(cfg.beam_size, pad=0, bos=1, eos=2, cuda=cfg.cuda)
        h       = h.expand(beam.size, h.size(1))
        c       = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))
        for _ in range(cfg.max_len):
            x = beam.get_current_state()
            o, h, c = self.decoder(x.unsqueeze(0), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h = h.index_select(0, beam.get_current_origin())
            c = c.index_select(0, beam.get_current_origin())
        return beam.get_hyp(0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c     = {p: i for i, p in enumerate(p_vocab)}
    c1 = "".join(chr(p2c[p]) for p in p_seq1)
    c2 = "".join(chr(p2c[p]) for p in p_seq2)
    return Levenshtein.distance(c1, c2) / len(c2)


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_test(model, test_data_raw, src_stoi, tgt_itos, config, batch_size=128):
    """
    Batches the encoder for speed, then runs beam search per example.
    """
    model.eval()
    total_per = total_wer = n = 0
    pad_idx = src_stoi["<pad>"]

    with torch.no_grad():
        for i in range(0, len(test_data_raw), batch_size):
            chunk = test_data_raw[i:i + batch_size]
            words, phonemes_batch = zip(*chunk)

            # batch encode all words and pad
            encoded = [torch.tensor(encode(list(w), src_stoi), dtype=torch.long)
                       for w in words]
            src = pad_sequence(encoded, batch_first=False,
                               padding_value=pad_idx)  # (seq, batch)
            if config.cuda:
                src = src.cuda()

            # run encoder once over the whole batch
            enc_out, h, c = model.encoder(src)  # enc_out: (seq, batch, hidden)

            # decode each example individually with beam search
            for j in range(len(chunk)):
                hj       = h[j].unsqueeze(0)                         # (1, hidden)
                cj       = c[j].unsqueeze(0)
                ctx_j    = enc_out[:, j, :].unsqueeze(0).transpose(0, 1)  # (1, seq, hidden) -- wait, enc_out is (seq, batch, hidden)
                # enc_out[:, j, :] is (seq, hidden), unsqueeze(0) → (1, seq, hidden)
                ctx_j    = enc_out[:, j, :].unsqueeze(0)             # (1, seq, hidden)

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model",      default="best_model.pt")
    cli.add_argument("--data_path",  default="../data/cmudict/")
    cli.add_argument("--seed",       type=int, default=5)
    cli.add_argument("--n_examples", type=int, default=10)
    cli.add_argument("--batch_size", type=int, default=128)
    cli.add_argument("--beam_size",  type=int, default=None,
                     help="Override beam size from checkpoint (1 = greedy, fastest)")
    args = cli.parse_args()

    # load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)    
    config      = checkpoint["config"]
    config.cuda = config.cuda and torch.cuda.is_available()
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

    # rebuild test split with same seed as training
    filepath      = os.path.join(args.data_path, "cmudict.dict")
    test_data_raw = load_cmudict(filepath, seed=args.seed)

    print(f"Evaluating {len(test_data_raw)} examples "
          f"(beam_size={config.beam_size}, batch_size={args.batch_size})...")
    run_test(model, test_data_raw, src_stoi, tgt_itos, config,
             batch_size=args.batch_size)

    print(f"\nShowing {args.n_examples} examples  (> word  = true  < predicted)\n")
    for i, (word, phonemes) in enumerate(test_data_raw):
        show(word, phonemes, model, src_stoi, tgt_itos, config)
        if i + 1 == args.n_examples:
            break