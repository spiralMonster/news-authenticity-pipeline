import tensorflow as tf
from tensorflow.keras.layers import Layer



@tf.keras.utils.register_keras_serializable()
class InputDividerLayer(Layer):
    """
    The layer to divide input so that it is fed to different LSTM networks.
    """
    def __init__(self,num_models,**kwargs):
        super().__init__(**kwargs)

        self.num_models=num_models
        self.supports_masking=True



    def call(self,x):
        shape=tf.shape(x)

        words_per_text=shape[1]
        embedding_dim=shape[2]

        tf.debugging.assert_equal(
            words_per_text % self.num_models,
            0,
            message="Sequence length must be divisible by num_models"
        )

        factor = words_per_text // self.num_models

        inputs=tf.cast(x,tf.float32)
        inputs=tf.reshape(inputs,[-1,self.num_models,factor,embedding_dim])
        inputs=tf.unstack(inputs,axis=1)

        return inputs

    def compute_mask(self,inputs,mask=None):
        if mask is None:
            return None

        shape=tf.shape(mask)
        words_per_text=shape[1]
        factor=words_per_text//self.num_models

        reshaped_mask=tf.reshape(mask,[-1,self.num_models,factor])

        final_mask=tf.unstack(reshaped_mask,axis=1)

        return final_mask


    def compute_output_shape(self,input_shape):
        words_per_text = input_shape[1]
        factor = words_per_text // self.num_models
        shape=[(input_shape[0],factor,input_shape[2]) for _ in range(self.num_models)]

        return shape

    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "num_models":self.num_models
            }
        )

        return config



