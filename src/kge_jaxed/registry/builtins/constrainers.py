"""Built-in constrainer registrations."""


def register_constrainers() -> None:
    from kge_jaxed.constraints.constrainers import (
        clip,
        max_norm,
        non_negative,
        unit_modulus,
        unit_norm,
    )
    from kge_jaxed.registry.core import constrainers

    constrainers.register("unit_norm", unit_norm, aliases=["normalize"])
    constrainers.register("max_norm", max_norm, aliases=["clamp_norm"])
    constrainers.register("clip", clip, aliases=["clamp"])
    constrainers.register("non_negative", non_negative)
    constrainers.register("unit_modulus", unit_modulus, aliases=["complex_normalize"])
