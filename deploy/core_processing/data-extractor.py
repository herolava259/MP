#from __future__ import absolute_import
import tensorflow as tf
import nibabel as nib
import glob
import numpy as np
import tarfile

IMG_FEATURE_KEYS = ['flair', 't1', 't1ce', 't2']
MSK_FEATURE_KEY  = 'mask'
def _function_wrapper(args_tuple):
    function, args = args_tuple

    return function(*args)

input_spec = {
    **{key: tf.io.FixedLenFeature([], tf.string)
       for key in IMG_FEATURE_KEYS},
    'index': tf.io.FixedLenFeature([], tf.int16)
}

_feature_spec = {
    **{key: tf.io.FixedLenFeature([], tf.string)
       for key in IMG_FEATURE_KEYS},
    'mask': tf.io.FixedLenFeature([], tf.string),
}


def extract_to_np_arr(niib_file, data_dir):

    offset = len('BraTS2021_')
    flair_f = glob.glob(data_dir + '*_flair.nii.gz')[0]
    t1_f = glob.glob(data_dir + f'*_t1.nii.gz')[0]
    t1ce_f = glob.glob(data_dir + f'*_t1ce.nii.gz')[0]
    t2_f = glob.glob(data_dir + f'*_t2.nii.gz')[0]

    flair_np = nib.load(flair_f).get_fdata()
    t1_np = nib.load(t1_f).get_fdata()
    t1ce_np = nib.load(t1ce_f).get_fdata()
    t2_np = nib.load(t2_f).get_fdata()

    return {
        'name': niib_file[offset:-1],
        'flair': flair_np,
        't1': t1_np,
        't1ce': t1ce_np,
        't2': t2_np
    }

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def concatenate_to_4d_array(dic):
    return np.concatenate(dic.values(), axis=0)

def save_to_tfrecords_file(img_dic, data_dir, writer):
    name = img_dic.pop('name')
    file_path = f'{data_dir}{name}.tfrecords'

    feature = {
        **{
            key: _bytes_feature(value)
            for (key,value) in img_dic.items()
        },
        'index': int(name)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    with tf.io.TFRecordWriter(file_path) as writer:
        writer.write(example.SerializeToString())




