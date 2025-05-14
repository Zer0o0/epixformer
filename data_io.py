import pickle
import random

import numpy as np
import tensorflow as tf
import tensorflow_text as text

from utils import dna_2onehot

# Write tfrecord data
# The following functions can be used to convert a value to a type compatible with tf.train.Example.
def _bytes_feature(value):
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(seq_dna, seq_epi, label):
    # Creates a tf.train.Example message ready to be written to a file.
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        'dna_seq': _bytes_feature(tf.io.serialize_tensor(
            tf.convert_to_tensor(dna_2onehot(seq_dna), dtype=tf.float32))),  # convert np.array into byte-string
        'epi_seq': _bytes_feature(seq_epi.encode('utf-8')),
        'label': _int64_feature(label),
    }

    # Create a Features message using tf.train.Example.
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(examples, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in examples:
            example = serialize_example(ex[0][0], ex[0][1], ex[1])
            writer.write(example)


# Parse tfrecord data
feature_description = {
    'dna_seq': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'epi_seq': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _parse_function(example):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example, feature_description)
    dna = tf.io.parse_tensor(parsed_example['dna_seq'], out_type=tf.float32)
    epi = parsed_example['epi_seq']
    label = parsed_example['label']
    return (dna, epi), label


def select_sample(sample_file, num=1e+6):
    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)  # [((dna_seq,epi_seq),label),...]
    if len(samples) <= num:
        samples_in = samples
    else:
        samples_in = random.sample(samples, num)
    return samples_in


def separate_dataset(dataset, frac_train=0.8, frac_test=0.2):
    dataset = np.asarray(dataset, dtype=object)
    num_train = int(len(dataset)*frac_train)
    idx = np.array([i for i in range(len(dataset))])
    idx_train = np.random.choice(idx, num_train, replace=False)
    idx_test = np.delete(idx, idx_train)
    return dataset[idx_train].tolist(), dataset[idx_test].tolist()
