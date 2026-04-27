"""Built-in regularizer registrations."""


def register_regularizers() -> None:
    from kge_jaxed.registry.core import regularizers
    from kge_jaxed.regularization.lp import LpRegularizer
    from kge_jaxed.regularization.np import NpRegularizer

    regularizers.register("lp", LpRegularizer)
    regularizers.register("np", NpRegularizer, aliases=["powersum", "power_sum", "n3"])
