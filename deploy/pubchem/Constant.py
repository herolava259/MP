import tensorflow as tf

class CFG(object):
    FLAIR = 'flair'
    T1_WEIGHT = 't1'
    T1_CONTRAST = 't1ce'
    T2_WEIGHT = 't2'
    INDEX = 'index'
    INPUT_SPEC={
        INDEX: tf.io.FixedLenFeature([]. tf.int16),
        FLAIR: tf.io.FixedLenFeature([], tf.string),
        T1_WEIGHT: tf.io.FixedLenFeature([], tf.string),
        T1_CONTRAST: tf.io.FixedLenFeature([], tf.string),
        T2_WEIGHT: tf.io.FixedLenFeature([], tf.string),
    }