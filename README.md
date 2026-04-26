# KGE-JAXed

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![JAX](https://custom-icon-badges.demolab.com/badge/JAX-222827?logo=jax&logoColor=ffffff)](#)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This project started as a way for me to learn JAX by building something I already understood: Knowledge Graph Embedding models.

It is not intended to be a production-ready knowledge graph embedding library, and I would not recommend it as the default choice for serious work. If you want a mature package with broader coverage, better tooling, and a much more complete feature set, use [PyKEEN](https://github.com/pykeen/pykeen).

KGE-JAXed is a small JAX/Flax NNX knowledge graph embedding library inspired by PyKEEN. The main goal is to keep the code easy to read and easy to extend while still following JAX-friendly patterns.

## Installation

This repository uses `uv`.

```bash
uv sync
```

## What this project is

At the moment, this is mainly a small package for:

- A simple training and evaluation pipeline via `KGEPipeline`
- Trying out a few classic KGE models in JAX
- Running experiments on datasets provided by PyKEEN
- Learning and experimenting, rather than building a full production library

If you just want to use it, the main thing you need is the pipeline.

## Quick start

The simplest way to train a model is:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="transe",
    dataset="nations",
    loss_name="mrl",
    embedding_dim=128,
    negative_samples=1,
    learning_rate=1e-2,
    optimizer_name="adam",
    seed=42,
)

pipeline.train(
    epochs=100,
    log_every=10,
)
```

That will train a `TransE` model on the PyKEEN `nations` dataset.

## Recommended first runs

- Start with `transe` + `mrl` on `nations` if you want the quickest first run
- Try `rotate` + `nssa` once you want something a bit heavier
- Use smaller embedding sizes and fewer epochs first if you are just checking that everything works

## Train and evaluate a model

Here is a fuller example that trains a model and then evaluates it on the test split:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="transe",
    dataset="nations",
    loss_name="mrl",
    embedding_dim=128,
    negative_samples=1,
    learning_rate=1e-2,
    optimizer_name="adam",
    seed=42,
)

train_summary = pipeline.train(
    epochs=100,
    log_every=10,
)

metrics_df, ranks_df = pipeline.evaluate(
    split="test",
    filtered=True,
)

print(train_summary["train_losses"][-5:])
print(metrics_df)
```

`metrics_df` gives you the ranking metrics such as MRR, MR, and Hits@K.

`ranks_df` is the more interesting output if you want to inspect model behaviour in detail. Instead of only giving you aggregate metrics, it gives you a row-by-row view of evaluation results for each triple.

It contains:

- `head`, `relation`, `tail` for the evaluated triple
- `rank_head` for head prediction rank
- `rank_tail` for tail prediction rank
- `score_head` for the score assigned to the true head query
- `score_tail` for the score assigned to the true tail query

That makes it easy to answer questions like:

- Which triples are hardest for the model?
- Is the model much better at head prediction or tail prediction?
- Which examples are getting rank 1 and which are failing badly?
- Which relations seem to be causing the worst errors?

For example:

```python
print(ranks_df.head())

# Hardest triples by tail rank
print(ranks_df.sort_values("rank_tail", ascending=False).head(10))

# Hardest triples by head rank
print(ranks_df.sort_values("rank_head", ascending=False).head(10))
```

I like this because it makes evaluation much easier to debug. You can look beyond a single MRR number and actually inspect where the model is doing well or badly.

A typical `metrics_df` looks like this:

```text
         head   tail    avg
mrr      ...    ...    ...
mr       ...    ...    ...
hits@1   ...    ...    ...
hits@3   ...    ...    ...
hits@10  ...    ...    ...
```

And `ranks_df` looks like this:

```text
   head  relation  tail  rank_head  rank_tail  score_head  score_tail
0     0         1     2          1          3      ...        ...
1     4         2     7          8          1      ...        ...
```

## Train with a different model

You can swap the model and loss configuration directly in the pipeline:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="rotate",
    dataset="fb15k",
    loss_name="nssa",
    embedding_dim=256,
    negative_samples=16,
    learning_rate=1e-3,
    optimizer_name="adam",
    dataset_kwargs={
        "batch_size": 256,
        "shuffle": True,
    },
    loss_kwargs={
        "adversarial_temperature": 1.0,
        "margin": 6.0,
    },
    seed=42,
)

pipeline.train(
    epochs=50,
    log_every=5,
)

metrics_df, ranks_df = pipeline.evaluate(
    split="test",
    filtered=True,
    ks=(1, 3, 10),
)

print(metrics_df)
```

## Embedding initializers, constrainers, and regularizers

KGE models often need small operations around the embedding tables in addition to
the interaction function itself.

- Initializers choose the starting embedding values before training.
- Constrainers project embedding values after initialization and after each
  optimizer step. They enforce hard constraints such as unit-length entity
  vectors or unit-modulus complex relation values.
- Regularizers add a soft penalty to the training loss. They encourage smaller
  or simpler embeddings, but they do not strictly enforce a constraint.

These are configured separately for entity and relation embeddings through
`model_kwargs`:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="rotate",
    dataset="nations",
    loss_name="nssa",
    embedding_dim=128,
    model_kwargs={
        "entity_embedding_kwargs": {
            "embedding_init": "normal_norm",
            "embedding_init_kwargs": {"stddev": 0.1},
        },
        "relation_embedding_kwargs": {
            "embedding_init": "init_phases",
        },
        "relation_constrainer_kwargs": {
            "name": "unit_modulus",
        },
        "entity_regularizer_kwargs": {
            "name": "lp",
            "p": 2.0,
            "normalize": True,
            "weight": 0.01,
        },
    },
)
```

The most useful built-in initializer names are:

| Name | Meaning |
| --- | --- |
| `uniform`, `normal` | Random uniform or normal initialization |
| `uniform_norm`, `normal_norm` | Random initialization followed by row-wise unit normalization |
| `xavier`, `xavier_uniform`, `glorot_uniform` | Glorot uniform initialization |
| `xavier_uniform_norm`, `glorot_uniform_norm` | Glorot uniform initialization followed by row-wise unit normalization |
| `xavier_normal`, `glorot_normal` | Glorot normal initialization |
| `xavier_normal_norm`, `glorot_normal_norm` | Glorot normal initialization followed by row-wise unit normalization |
| `complex_normal`, `complex_uniform` | Complex-valued initialization with independent real and imaginary parts |
| `init_phases`, `complex_phases` | Complex unit-modulus values, useful for RotatE relations |

The built-in constrainers are:

| Name | Meaning |
| --- | --- |
| `unit_norm` or `normalize` | Normalize each embedding row to unit L2 norm |
| `max_norm` or `clamp_norm` | Project rows with norm above `max_value` back to that norm |
| `clip` or `clamp` | Clip individual values into `[min_value, max_value]` |
| `non_negative` | Replace negative values with zero |
| `unit_modulus` or `complex_normalize` | Project each complex value to magnitude one |

The built-in regularizers are:

| Name | Meaning |
| --- | --- |
| `lp` | Mean or sum of row-wise Lp norms |
| `np`, `powersum`, `power_sum`, `n3` | Mean or sum of `sum(abs(x) ** p)` per row; `p=3` gives an N3-style penalty |

Passing `None` for a model config uses that model's default. Passing an empty
dict such as `entity_constrainer_kwargs={}` disables that default. This matters
for models such as `TransE`, which constrains entity embeddings by default, and
`RotatE`, which constrains relation embeddings to unit modulus by default.

## Save and resume training

You can also save checkpoints during training:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="transe",
    dataset="nations",
    loss_name="mrl",
    embedding_dim=128,
)

pipeline.train(
    epochs=50,
    log_every=5,
    save_checkpoint_dir="checkpoints/transe-nations",
    save_every=10,
)
```

Then later:

```python
from kge_jaxed import KGEPipeline

pipeline = KGEPipeline(
    model="transe",
    dataset="nations",
    loss_name="mrl",
    embedding_dim=128,
)

pipeline.load_checkpoint("checkpoints/transe-nations")
pipeline.train(epochs=20, log_every=5)
```

## Datasets

This project uses PyKEEN datasets rather than re-implementing dataset download and packaging.

If you pass a dataset name such as `"nations"` or `"fb15k"` into the pipeline, it is resolved through PyKEEN. For the full list of available datasets, see the [PyKEEN dataset documentation](https://pykeen.readthedocs.io/en/latest/reference/datasets.html).

## Supported models

These are the models currently registered in the library:

| Model name | Class | Status |
| --- | --- | --- |
| `transe` | `TransE` | Implemented |
| `distmult` | `DistMult` | Implemented |
| `complex` | `ComplEx` | Implemented |
| `rotate` | `RotatE` | Implemented |

## Supported losses

| Loss name | Meaning |
| --- | --- |
| `mrl` | Margin ranking loss |
| `bce` | Binary cross-entropy loss |
| `softplus` | Softplus loss |
| `nssa` | Self-adversarial negative sampling loss |

## Current scope

This repository is best thought of as:

- A personal JAX learning project
- A clean, small reference implementation
- A place to experiment with KGE models in JAX and Flax NNX
- Not a polished end-user library
- Intentionally small in model coverage

If you need a broader and more battle-tested package, PyKEEN is the better choice.

## Development

Linting and unit tests are configured through `tox`.

```bash
tox -e lint
tox -e test
```
