"""Built-in negative sampler registrations."""


def register_samplers() -> None:
    from kge_jaxed.negative_sampling.uniform_negative_sampling import uniform_balanced_sampler
    from kge_jaxed.registry.core import negative_samplers

    negative_samplers.register("uniform", uniform_balanced_sampler)
