from typing_extensions import List,Tuple

from tfx.components import SchemaGen
from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater

updating_params = [
    ("text", 0.85),
    ("label", 0.85),
    ("num_single_quote_error", 0.85),
    ("num_spacing_error", 0.85),
    ("num_social_media_handles", 0.85),
    ("num_urls", 0.85),
    ("num_twitter_post_urls", 0.85),
    ("num_hashtags", 0.85),
    ("num_space_absence_after_sentence_completion", 0.85),
    ("num_capitalized_words", 0.85),
    ("num_capitalization_absence_after_sentence_completion", 0.85),
    ("num_spelling_errors", 0.85),

]


def UpdateSchema(schema_gen:SchemaGen,updating_params:List[Tuple[str,float]]=updating_params):
    schema_updater=SchemaUpdater(
        schema=schema_gen.outputs["schema"],
        updating_params=updating_params
    )

    print(f"[INFO] Schema Updated.")

    return schema_updater
