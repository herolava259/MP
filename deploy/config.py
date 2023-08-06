import os

PROJECT_NAME = "medical_image_segmentation/"

TRAIN_RECORD_FILE = 'train.tfrecord'

PIPELINE_ROOT = os.path.join(PROJECT_NAME, 'pipeline/')

_DATA_ROOT = os.path.join(PROJECT_NAME, 'data/')

_DATA_TRAIN_FILE_PATH = os.path.join(_DATA_ROOT, TRAIN_RECORD_FILE)


METADATA_PATH = os.path.join('metadata', PROJECT_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model', PROJECT_NAME)

from absl import logging

logging