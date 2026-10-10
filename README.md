# KGE-JAXed

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![JAX](https://custom-icon-badges.demolab.com/badge/JAX-222827?logo=jax&logoColor=ffffff)](#)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A small knowledge graph embedding (KGE) library built on JAX and Flax NNX, inspired by [PyKEEN](https://github.com/pykeen/pykeen).

I started this to learn JAX by building something I already understood. That shaped the design: the code is meant to be read, so I favour explicit, simple implementations over clever abstractions, and get speed from JAX patterns (`jit`, `vmap`) rather than complexity.

> This is a learning project, not a production library. If you need something mature, well tested, and feature complete, use PyKEEN.

## What it does

- Trains `TransE`, `DistMult`, `ComplEx`, `RotatE` and `TranS`
- Evaluates link prediction with MR, MRR and Hits@K
- Returns per-triple ranks and scores, not just aggregate metrics
- Saves and resumes training from checkpoints
- Lets you configure embedding initializers, constrainers and regularizers

Everything goes through `KGEPipeline`, which handles the dataset, model, training, evaluation and checkpointing. Datasets are loaded through PyKEEN, so this project only has to deal with models, losses, sampling and evaluation.

## Installation

KGE-JAXed needs Python 3.12 or newer and uses [uv](https://docs.astral.sh/uv/) to manage the environment. If you don't have uv yet, follow its [install guide](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root run:

```bash
uv sync
```

This creates a `.venv` in the project and installs the locked dependencies from `uv.lock`. The repo's `.python-version` pins Python 3.12, so uv will download that interpreter if you don't already have one.

You don't need to activate the environment. Prefix commands with `uv run` and uv uses the project's `.venv`:

```bash
uv run python my_script.py
```

If you prefer an activated shell, run `source .venv/bin/activate` instead. `ipykernel` is installed too, so you can pick the `.venv` as a kernel in Jupyter or VS Code notebooks.

### Hardware support

Which JAX build you get depends on your platform (see `pyproject.toml`):

| Platform | JAX install | Runs on |
| --- | --- | --- |
| Linux, x86_64 | `jax[cuda13]` | NVIDIA GPU (CUDA 13) |
| Linux, other architectures (e.g. ARM) | `jax` | CPU |
| macOS | `jax` | CPU |

The CUDA build needs a recent NVIDIA driver. Without a compatible GPU, JAX falls back to the CPU. Nothing in the library is CUDA-specific, so the code is the same everywhere. The small datasets such as `nations` train fine on a laptop CPU, but larger ones like `fb15k` will be much slower without a GPU.

## Quick start

Train a TransE model on the PyKEEN `nations` dataset, then evaluate it on the test split:

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

train_summary = pipeline.train(epochs=100, log_every=10)
metrics_df, ranks_df = pipeline.evaluate(split="test", filtered=True)

print(train_summary["train_losses"][-5:])
print(metrics_df)
```

`evaluate()` returns two dataframes. `metrics_df` holds the aggregate metrics, split by head and tail prediction:

```text
         head   tail    avg
mrr      ...    ...    ...
mr       ...    ...    ...
hits@1   ...    ...    ...
hits@3   ...    ...    ...
hits@10  ...    ...    ...
```

`ranks_df` has one row per evaluated triple (see the next section).

## Looking at individual triples

A single MRR number tells you little about where a model fails. `ranks_df` keeps the per-triple detail:

```text
   head  relation  tail  rank_head  rank_tail  score_head  score_tail
0     0         1     2          1          3      ...        ...
1     4         2     7          8          1      ...        ...
```

`rank_head` and `rank_tail` are the ranks of the true head and tail, and `score_head` and `score_tail` are the scores the model gave them.

To make it readable, map the ids back to labels:

```python
ranks_with_labels = ranks_df.assign(
    head_label=lambda df: df["head"].map(pipeline.dataset.id_to_entity),
    relation_label=lambda df: df["relation"].map(pipeline.dataset.id_to_relation),
    tail_label=lambda df: df["tail"].map(pipeline.dataset.id_to_entity),
)
```

Then ask questions of it. For example, the triples with the worst tail rank:

```python
print(
    ranks_with_labels.sort_values("rank_tail", ascending=False)[
        ["head_label", "relation_label", "tail_label", "rank_tail", "score_tail"]
    ].head(10)
)
```

Or which relations are hardest on average:

```python
relation_summary = (
    ranks_with_labels.assign(avg_rank=lambda df: (df["rank_head"] + df["rank_tail"]) / 2)
    .groupby("relation_label")["avg_rank"]
    .mean()
    .sort_values(ascending=False)
)

print(relation_summary.head(10))
```

## Making predictions

Once you have a checkpoint, `KGEPredict` scores every entity for a query such as `(usa, treaties, ?)`:

```python
from kge_jaxed import KGEPredict

predictor = KGEPredict("checkpoints/transe-nations")

tail_scores = predictor.predict("usa", "treaties", predict_side="tail")
print(tail_scores.head(10))
```

You pass entity and relation labels, not ids. `predict_side` is `"tail"` to complete `(head, relation, ?)` or `"head"` to complete `(?, relation, tail)`. In the head case, the entity you pass is the tail.

The result is a dataframe sorted by score, highest first (higher means more plausible), with one row per entity:

```text
    entity  score  in_train  in_test  in_val
0   ...     ...    True      False    False
1   ...     ...    False     True     False
```

The `in_train`, `in_test` and `in_val` columns tell you whether the completed triple is already in that split. This helps you separate facts the model has seen from genuinely new predictions. Filtering to `~in_train` is a good way to find candidates for missing links.

`KGEPredict` rebuilds the dataset from the name stored in the checkpoint metadata. It loads the dataset with default settings and ignores any `dataset_kwargs` you trained with. The entity and relation labels come from that dataset (see `pipeline.dataset.entity_to_id` and `relation_to_id`).

## Using a different model

Models, losses and their options are all set on the pipeline:

```python
pipeline = KGEPipeline(
    model="rotate",
    dataset="fb15k",
    loss_name="nssa",
    embedding_dim=256,
    negative_samples=16,
    learning_rate=1e-3,
    optimizer_name="adam",
    dataset_kwargs={"batch_size": 256, "shuffle": True},
    loss_kwargs={"adversarial_temperature": 1.0, "margin": 6.0},
    seed=42,
)

pipeline.train(epochs=50, log_every=5)
metrics_df, ranks_df = pipeline.evaluate(split="test", filtered=True, ks=(1, 3, 10))
```

## Saving and resuming

Pass a directory to `train()` to write checkpoints as you go:

```python
pipeline = KGEPipeline(model="transe", dataset="nations", loss_name="mrl", embedding_dim=128)

pipeline.train(
    epochs=50,
    log_every=5,
    save_checkpoint_dir="checkpoints/transe-nations",
    save_every=10,
)
```

Later, build a pipeline with the same configuration and load the checkpoint to carry on:

```python
pipeline = KGEPipeline(model="transe", dataset="nations", loss_name="mrl", embedding_dim=128)

pipeline.load_checkpoint("checkpoints/transe-nations")
pipeline.train(epochs=20, log_every=5)
```

## Datasets

Datasets are resolved through PyKEEN, so any name it knows (`"nations"`, `"fb15k"`, ...) works. See the [PyKEEN dataset docs](https://pykeen.readthedocs.io/en/latest/reference/datasets.html) for the full list.

## Models

| Name | Class |
| --- | --- |
| `transe` | `TransE` |
| `trans` | `TranS` |
| `distmult` | `DistMult` |
| `complex` | `ComplEx` |
| `rotate` | `RotatE` |

## Losses

| Name | Loss |
| --- | --- |
| `mrl` | Margin ranking loss |
| `bce` | Binary cross-entropy |
| `softplus` | Softplus |
| `nssa` | Self-adversarial negative sampling |

## Initializers, constrainers and regularizers

Most KGE models need a bit more than an interaction function around their embedding tables:

- **Initializers** set the starting embedding values.
- **Constrainers** are hard constraints, applied after initialization and after every optimizer step (for example, unit-length entity vectors).
- **Regularizers** are soft penalties added to the training loss.

Each is configured separately for entities and relations through `model_kwargs`:

```python
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
        "relation_embedding_kwargs": {"embedding_init": "init_phases"},
        "relation_constrainer_kwargs": {"name": "unit_modulus"},
        "entity_regularizer_kwargs": {"name": "lp", "p": 2.0, "normalize": True, "weight": 0.01},
    },
)
```

Each model has sensible defaults. Embedding kwargs you pass are merged over those defaults, while constrainer and regularizer kwargs replace them. An empty dict switches a default off, e.g. `entity_constrainer_kwargs={}` stops `TransE` constraining its entity embeddings (it does by default). `RotatE` constrains its relations to unit modulus by default.

**Initializers**

| Name | Meaning |
| --- | --- |
| `uniform`, `normal` | Random uniform or normal |
| `uniform_norm`, `normal_norm` | As above, then row-wise unit normalization |
| `xavier`, `xavier_uniform`, `glorot_uniform` | Glorot uniform |
| `xavier_uniform_norm`, `glorot_uniform_norm` | Glorot uniform, then row-wise unit normalization |
| `xavier_normal`, `glorot_normal` | Glorot normal |
| `xavier_normal_norm`, `glorot_normal_norm` | Glorot normal, then row-wise unit normalization |
| `complex_normal`, `complex_uniform` | Complex values with independent real and imaginary parts |
| `init_phases`, `complex_phases` | Complex values of unit modulus, useful for RotatE relations |

**Constrainers**

| Name | Meaning |
| --- | --- |
| `unit_norm`, `normalize` | Normalize each row to unit L2 norm |
| `max_norm`, `clamp_norm` | Project rows with norm above `max_value` back to that norm |
| `clip`, `clamp` | Clip values into `[min_value, max_value]` |
| `non_negative` | Replace negative values with zero |
| `unit_modulus`, `complex_normalize` | Project each complex value to magnitude one |

**Regularizers**

| Name | Meaning |
| --- | --- |
| `lp` | Mean or sum of row-wise Lp norms |
| `np`, `powersum`, `power_sum`, `n3` | Mean or sum of `sum(abs(x) ** p)` per row; `p=3` gives an N3-style penalty |

## Project layout

The code is meant to be read top to bottom, starting at `KGEPipeline` in `src/kge_jaxed/pipeline.py`. It owns the dataset, model, optimizer and the training loop, and it calls into everything else:

```text
src/kge_jaxed/
├── pipeline.py          # KGEPipeline: train, evaluate, checkpoint (start here)
├── predict.py           # KGEPredict: score queries from a saved checkpoint
├── rngs.py              # Explicit RNG handling, derived from a single seed
├── models/              # TransE, DistMult, ComplEx, RotatE, TranS, plus embeddings and initializers
├── loss_functions/      # mrl, bce, softplus, nssa
├── negative_sampling/   # Negative samplers
├── constraints/         # Constrainers
├── regularization/      # Regularizers
├── training/            # JIT-compiled train step, setup and checkpointing
├── evaluation/          # Ranking and metrics
├── datasets/            # Dataset wrappers, including PyKEEN
└── registry/            # Maps string names ("transe", "mrl", ...) to classes
```

A model only has to implement `interaction_function(h, r, t)`. Scores follow the convention that higher means more plausible, and every built-in loss is written against that.

## Development

Tests and linting run through `tox`:

```bash
tox -e test
tox -e lint
```

## Licence

MIT. See [LICENSE](LICENSE).
