# g2p-pytorch

A Grapheme-to-Phoneme (G2P) model in PyTorch — converts written words to
[ARPAbet](https://en.wikipedia.org/wiki/ARPABET) phonemes, including words
not in any dictionary.

Built for use in TTS pipelines. Updated from the original
[2017 tutorial](https://fehiepsi.github.io/blog/grapheme-to-phoneme.html)
to modern PyTorch with no torchtext dependency.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python infer.py hello
# hello -> HH AH0 L OW1

python infer.py --checkpoint best_model.pt psychology
```

### As a plugin / import

```python
from infer import G2PInference

g2p = G2PInference("best_model.pt")   # load once at startup

# single word
g2p("hello")
# → ['HH', 'AH0', 'L', 'OW1']

# works on unfamiliar/made-up words too
g2p("ghiblification")
# → ['G', 'IH0', 'B', 'L', 'IH0', 'F', 'IH0', 'K', 'EY1', 'SH', 'AH0', 'N']
```

Phonemes follow the [ARPAbet](https://en.wikipedia.org/wiki/ARPABET) standard
used by CMUDict. Vowels have stress markers (0 = unstressed, 1 = primary, 2 = secondary).

### Using in a TTS pipeline

```python
from infer import G2PInference

class MyTTS:
    def __init__(self):
        self.g2p = G2PInference("best_model.pt")  # load once

    def text_to_phonemes(self, text: str) -> list:
        return [self.g2p(word) for word in text.lower().split()]
```

## Getting a trained model

You need a `best_model.pt` checkpoint to run inference. Train your own (requires a GPU for reasonable speed):

```bash
python train.py
# saves best_model.pt automatically
```

## Training

CMUDict (~134k words) is downloaded automatically on first run. Training stops via early stopping when validation loss plateaus.

```bash
python train.py
```

Common options:
```bash
python train.py --epochs 100           # train longer
python train.py --d_hidden 1024        # bigger model
python train.py --no_attention         # disable attention
python train.py --no_cuda              # force CPU (slow)
```

Expected results after full training (~30–50 epochs on GPU):

| Metric | Value |
|--------|-------|
| Phoneme Error Rate (PER) | ~9.8% |
| Word Error Rate (WER) | ~40.7% |

## Evaluation

```bash
python g2p.py --test
```

## Why neural G2P?

Dictionary-based approaches (like just looking up CMUDict) fail on:
- Names and proper nouns ("Nguyen", "Saoirse")
- Technical or scientific terms
- Made-up words, brand names, neologisms

A trained G2P model learns spelling-to-sound rules from data and generalizes
them to words it has never seen.

## Project structure

```
neuralg2p/
├── __init__.py     exports public API
├── model.py        Encoder, Attention, Decoder, G2P, Beam
├── data.py         load_cmudict, Vocab, encode, collate_fn
├── metrics.py      phoneme_error_rate (Levenshtein-based)
├── infer.py        G2PInference — load checkpoint and predict
├── train.py        training script
└── g2p.py          monolithic script (train + evaluate in one file)
```

## Credits

- Original tutorial: [fehiepsi](https://fehiepsi.github.io/blog/grapheme-to-phoneme.html)
- Encoder/Decoder: [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- Beam search: [Seq2Seq-PyTorch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch/)
- PER metric: [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
