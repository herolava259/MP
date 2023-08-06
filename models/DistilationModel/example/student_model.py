from tensorflow import keras
import tensorflow as tf

# Student model
def create_small_model():
  tf.random.set_seed(42)
  model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(2)
  ])

  return model