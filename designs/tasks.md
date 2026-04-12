# Tasks 📚

Tasks define what we train and evaluate the transformer blocks on.

Since BlockBuster tests only **a few transformer layers** (not full models), tasks must:
- Load fast and stay small
- Produce a meaningful loss signal from shallow block stacks
- Map cleanly to the metrics in [metrics.md](./metrics.md)

The primary task type is **Causal Language Modeling (CLM)** — next-token prediction.
It exercises attention, residual connections, and norms all at once, making it the most direct probe for hack impact.

---

## Datasets

### Tier 1 — Tiny, Instant Iteration

| Dataset | Size | Load |
|---|---|---|
| **Shakespeare** | ~1 MB | manual download or `datasets` |
| **Penn Treebank (PTB)** | ~5 MB | `datasets` / `torchtext` |
| **WikiText-2** | ~12 MB | `datasets` |

These are the default datasets for hack development and rapid iteration.

#### WikiText-2 ⭐ (primary)

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
```

- Well-studied perplexity baseline — easy to sanity-check results
- Clean text, no noise
- Fast enough to run many hack comparisons in one session

#### Penn Treebank

```python
from datasets import load_dataset
ds = load_dataset("ptb_text_only")
```

- Even smaller than WT-2, different domain (newswire)
- Good second reference point for perplexity tax comparisons

#### Shakespeare

```python
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

- Minimal setup, good for sanity checks and debugging new hacks
- Works well at character or BPE token level

---

### Tier 2 — Richer Signal, Longer Runs

| Dataset | Size | Load |
|---|---|---|
| **WikiText-103** | ~500 MB | `datasets` |
| **TinyStories** | ~2 GB (streamable) | `datasets` |

Use these when a hack passes Tier 1 and needs a more demanding signal before being declared viable.

#### WikiText-103

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
```

- Same format as WT-2, drop-in replacement for longer runs
- More data = better gradient norm and spike rate statistics

#### TinyStories

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", streaming=True)
```

- Synthetic, simple vocabulary — ideal for watching a few blocks learn structure
- Stream it to avoid storing 2 GB locally

---

## Task Configuration

| Setting | Default |
|---|---|
| Sequence length | 512 |
| Tokenizer | GPT-2 BPE (`tiktoken` or `transformers`) |
| Batch size | fit to VRAM of baseline run |
| Primary metric | Perplexity (PPL) on validation split |

Sequence length of 512 is a reasonable default — long enough to stress KV cache and attention, short enough to iterate fast on Tier 1 datasets.
