from typing_extensions import List,Tuple

from tfx.components import SchemaGen
from components.custom_components.schema_updater.build.schema_updater_component import SchemaUpdater

updating_params = [
    ("text", 0.75),
    ("label", 0.75),
    ("num_single_quote_error", 0.75),
    ("num_spacing_error", 0.75),
    ("num_space_absence_after_sentence_completion", 0.75),
    ("num_capitalized_words", 0.75),
    ("num_capitalization_absence_after_sentence_completion", 0.75),
    ("num_spelling_errors", 0.75),
    ("num_words", 0.75),
    ("num_punctuations", 0.75),
    ("num_numeric_values", 0.75),

]

def UpdateSchema(schema_gen:SchemaGen,updating_params:List[Tuple[str,float]]=updating_params):
    schema_updater=SchemaUpdater(
        schema=schema_gen.outputs["schema"],
        updating_params=updating_params
    )

    print(f"[INFO] Schema Updated.")

    return schema_updater
