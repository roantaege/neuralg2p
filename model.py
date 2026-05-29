import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm      = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden  = d_hidden

    def forward(self, x_seq):
        # x_seq: (seq_len, batch)
        e_seq = self.embedding(x_seq)
        batch = x_seq.size(1)
        h = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        c = torch.zeros(batch, self.d_hidden, device=x_seq.device)
        outputs = []
        for e in e_seq.chunk(e_seq.size(0), dim=0):
            h, c = self.lstm(e.squeeze(0), (h, c))
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
        # x: (batch, dim)   context: (batch, seq, dim)
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
    """Ordered beam of candidate outputs.
    Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
    """
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
    # Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
