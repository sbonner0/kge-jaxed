"""Built-in model registrations."""


def register_models() -> None:
    from kge_jaxed.models.complex import ComplEx
    from kge_jaxed.models.distmult import DistMult
    from kge_jaxed.models.rotate import RotatE
    from kge_jaxed.models.transe import TransE
    from kge_jaxed.registry.core import models

    models.register("transe", TransE)
    models.register("distmult", DistMult)
    models.register("complex", ComplEx)
    models.register("rotate", RotatE)
