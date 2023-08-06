import os
os.environ['PYTHONHASHSEED']=str(42)

# Libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
from models.DistilationModel.example.format_input import format_image, BATCH_SIZE
from models.DistilationModel.example.student_model import create_small_model
# More random seed setup
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

#Define train/test splits
splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

# Download the dataset
(train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)

# Print useful information
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

print(f"There are {num_examples} images for {num_classes} classes.")

train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)
teacher = create_small_model()
teacher.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Notice from_logits param is set to True
    optimizer=keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Fit the model and save the training history (will take from 5 to 10 minutes depending on the GPU you were assigned to)
teacher_history = teacher.fit(train_batches, epochs=5, validation_data=validation_batches)