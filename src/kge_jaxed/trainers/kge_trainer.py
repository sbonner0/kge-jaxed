import optax
from flax import nnx


class KGETrainer:

    def __init__(self, model: nnx.Module) -> None:

        # TODO: think about loss function

        learning_rate = 0.005
        momentum = 0.9

        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

        nnx.display(optimizer)

    @nnx.jit
    def train_step(self, model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
        """Train for a single step."""

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
        optimizer.update(grads)  # In-place updates.

    def train(self):
        # Training loop - loop over the dataset objects here.
        pass
