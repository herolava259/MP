import tensorflow as tf
import nibabel as nib
import glob
import numpy as np
import tarfile
import os
IMG_FEATURE_KEYS = ['flair', 't1', 't1ce', 't2']
MSK_FEATURE_KEY = 'mask'

_feature_spec = {
    **{key: tf.io.FixedLenFeature([], tf.string)
       for key in IMG_FEATURE_KEYS},
    'mask': tf.io.FixedLenFeature([], tf.string),
}

def extract_to_np_arr(data_dir):

    flair_f = glob.glob(data_dir + '*_flair.nii.gz')[0]
    t1_f = glob.glob(data_dir + f'*_t1.nii.gz')[0]
    t1ce_f = glob.glob(data_dir + f'*_t1ce.nii.gz')[0]
    t2_f = glob.glob(data_dir + f'*_t2.nii.gz')[0]
    msk_f = glob.glob(data_dir + f'*_t2.nii.gz')[0]

    flair_np = nib.load(flair_f).get_fdata()
    t1_np = nib.load(t1_f).get_fdata()
    t1ce_np = nib.load(t1ce_f).get_fdata()
    t2_np = nib.load(t2_f).get_fdata()
    msk_np = nib.load(msk_f).get_fdata()

    return {
        'flair': flair_np,
        't1': t1_np,
        't1ce': t1ce_np,
        't2': t2_np,
        'msk': msk_np
    }

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
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

    writer.write(example.SerializeToString())


def fn1_test():
    file_path = 'C:/Users/Admin/Desktop/DATN_20222/seg_prj/FinalProject/data/raw/BraTS2021_00621.tar'

    os.mkdir('BraTS2021_00621')
    dest_dir = './BraTS2021_00621'
    with tarfile(file_path) as tar_zef:
        tar_zef.extractall(dest_dir)

    img_dic = extract_to_np_arr(dest_dir)

    print(img_dic['flair'].shape)

if __name__ == '__main__':
    fn1_test()