"""Built-in loss registrations."""

from functools import partial


def register_losses() -> None:
    from kge_jaxed.loss_functions.losses import (
        bce_loss,
        margin_ranking_loss,
        self_adversarial_negative_sampling_loss,
        softplus_loss,
    )
    from kge_jaxed.registry.core import losses

    losses.register("mrl", _loss_factory(margin_ranking_loss))
    losses.register("bce", _loss_factory(bce_loss))
    losses.register("softplus", _loss_factory(softplus_loss))
    losses.register("nssa", _loss_factory(self_adversarial_negative_sampling_loss))


def _loss_factory(loss_fn):
    def build_loss(**kwargs):
        if not kwargs:
            return loss_fn
        return partial(loss_fn, **kwargs)

    return build_loss
