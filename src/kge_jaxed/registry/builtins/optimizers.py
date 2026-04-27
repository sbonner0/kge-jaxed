"""Built-in optimizer registrations."""


def register_optimizers() -> None:
    import optax

    from kge_jaxed.registry.core import optimizers

    optimizers.register("adam", optax.adam)
    optimizers.register("adamw", optax.adamw)
    optimizers.register("sgd", optax.sgd)
    optimizers.register("adagrad", optax.adagrad)
    optimizers.register("rmsprop", optax.rmsprop)
    optimizers.register("adadelta", optax.adadelta)
    optimizers.register("adamax", optax.adamax)
