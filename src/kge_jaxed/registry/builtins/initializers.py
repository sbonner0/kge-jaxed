"""Built-in embedding initializer registrations."""


def register_initializers() -> None:
    from kge_jaxed.models import initializers as init
    from kge_jaxed.registry.core import initializers

    initializers.register("default", init._default_initializer)
    initializers.register("uniform", init._uniform_initializer)
    initializers.register("uniform_norm", init._uniform_norm_initializer)
    initializers.register("normal", init._normal_initializer)
    initializers.register("normal_norm", init._normal_norm_initializer)
    initializers.register("complex_normal", init._complex_normal_initializer)
    initializers.register(
        "xavier",
        init._xavier_uniform_initializer,
        aliases=["glorot", "xavier_uniform", "glorot_uniform"],
    )
    initializers.register(
        "xavier_uniform_norm",
        init._xavier_uniform_norm_initializer,
        aliases=["glorot_uniform_norm", "xavier_norm", "glorot_norm"],
    )
    initializers.register("xavier_normal", init._xavier_normal_initializer, aliases=["glorot_normal"])
    initializers.register("xavier_normal_norm", init._xavier_normal_norm_initializer, aliases=["glorot_normal_norm"])
    initializers.register("zeros", init._zeros_initializer)
    initializers.register("ones", init._ones_initializer)
    initializers.register("orthogonal", init._orthogonal_initializer)
    initializers.register("complex_uniform", init._complex_uniform_initializer)
    initializers.register("complex_phases", init._complex_phase_initializer, aliases=["init_phases", "phases"])
