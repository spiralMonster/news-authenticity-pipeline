from tfx.types import Channel
from tfx.types.standard_artifacts import Model,ModelBlessing

from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver



def ResolverNode():
    resolver=Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    )

    print(f"[INFO] Latest Blessed model has been resolved.")

    return resolver

