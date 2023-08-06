import os
os.environ['PYTHONHASHSEED']=str(42)

# Libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
from models.DistilationModel.example.format_input import format_image, BATCH_SIZE
from models.DistilationModel.example.teacher_model import create_big_model
from models.DistilationModel.example.student_model import create_small_model
from models.DistilationModel.DistillerModel import Distiller
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


teacher_weight_path = ''
student_weight_path = ''

def create_distiller(teacher_weight_path,
                     student_weight_path,
                     loss,
                     optimizer,
                     metrics,
                     alpha,
                     temperature):
    teacher = create_big_model()
    student = create_small_model()

    teacher.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Notice from_logits param is set to True
        optimizer=keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    student.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Notice from_logits param is set to True
        optimizer=keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    teacher.load_weights(teacher_weight_path)
    student.load_weights(student_weight_path)

    distiller = Distiller(student= student, teacher = teacher)

    distiller.compile(
        student_loss_fn=loss,
        optimizer=optimizer,
        metrics = metrics,
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha = alpha,
        temperature=temperature,
    )

    return distiller
distiller = create_distiller(teacher_weight_path,
                             student_weight_path,
                             loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             optimizer=keras.optimizers.Adam(),
                             metrics=[keras.metrics.SparseCategoricalAccuracy()],
                             distillation_loss_fn=keras.losses.KLDivergence(),
                             alpha=0.05,
                             temperature=5,
                             )
distiller_history = distiller.fit(train_batches, epochs=5, validation_data=validation_batches)

# Fit the model and save the training history (will take from 5 to 10 minutes depending on the GPU you were assigned to)
