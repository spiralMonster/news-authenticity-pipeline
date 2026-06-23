from datetime import datetime

from tfx.types import Channel
from tfx.types.standard_artifacts import Model,ModelBlessing

from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver



def ResolverNode():
    print(f"[{datetime.now()}] [START] Model Resolver Component.")

    resolver=Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    print(f"[{datetime.now()}] [END] Model Resolver Component.")

    return resolver

