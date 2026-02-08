import jax.numpy as jnp

from kge_jaxed.models.base_kge import BaseKGE


class DummyKGE(BaseKGE):
    def __init__(self, entity_w, relation_w, **kwargs):
        super().__init__(
            num_entities=1,
            num_relations=1,
            entity_embedding_dim=1,
            **kwargs,
        )
        self._entity_w = jnp.asarray(entity_w)
        self._relation_w = jnp.asarray(relation_w)

    def entity_weights(self):
        return self._entity_w

    def relation_weights(self):
        return self._relation_w

    def interaction_function(self, h, r, t):
        return jnp.zeros((h.shape[0],), dtype=h.dtype)


def test_entity_relation_regularization_loss():
    entity_w = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    relation_w = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    model = DummyKGE(
        entity_w,
        relation_w,
        entity_regularizer_kwargs={"name": "lp", "p": 1.0, "weight": 0.5},
        relation_regularizer_kwargs={"name": "lp", "p": 2.0, "weight": 2.0},
    )

    loss = float(model.regularization_loss())
    assert abs(loss - 6.5) < 1e-6
