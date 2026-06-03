import os
from typing_extensions import Text,Dict,List,Any

from tfx.components.base.base_executor import BaseExecutor
from tfx.types.standard_artifacts import Artifact
import tensorflow_data_validation as tfdv


class SchemaUpdaterExecutor(BaseExecutor):
    def Do(
            self,
            input_dict:Dict[Text,List[Artifact]],
            output_dict:Dict[Text,List[Artifact]],
            exec_properties: Dict[Text,Any]
    ):
        self._log_startup(input_dict,output_dict,exec_properties)

        schema_dir=input_dict["schema"][0].uri
        schema_location=os.path.join(schema_dir,os.listdir(schema_dir)[0])

        schema=tfdv.load_schema_text(schema_location)

        updating_params=exec_properties["updating_params"]
        for (feat,param_value) in updating_params:
            feature=tfdv.get_feature(schema,feat)
            feature.presence.min_fraction=param_value

        output_dir=output_dict["updated_schema"][0].uri
        tfdv.write_schema_text(
            schema,
            os.path.join(output_dir,"updated_schema.pbtxt")
        )
