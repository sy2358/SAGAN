from __future__ import print_function

import sox
from os import listdir
from os.path import isfile, join
import regex as re
import random
import argparse

import tensorflow as tf
import numpy as np
from collections import namedtuple, OrderedDict
from subprocess import call
import scipy.io.wavfile as wavfile
import codecs
import timeit
import struct
import toml
import sys
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encoder_proc(nn_file, ref_file, out_file, wav_canvas_size):
    """ Read and slice the wav and noisy files and write to TFRecords.
        out_file: TFRecordWriter.
    """

    fm, nn_data = wavfile.read(nn_file)
    fm, ref_data = wavfile.read(ref_file)

    max_size = len(nn_data)
    if len(ref_data) > max_size:
        max_size = len(ref_data)

    if max_size > wav_canvas_size:
        print("signal too long - skipping")
        return 0
    max_size = wav_canvas_size

    nn_data_pad = np.zeros(max_size, dtype=np.int32)
    nn_data_pad[:nn_data.shape[0]] = nn_data
    ref_data_pad = np.zeros(max_size, dtype=np.int32)
    ref_data_pad[:ref_data.shape[0]] = ref_data

    nn_raw = nn_data_pad.tostring()
    ref_raw = ref_data_pad.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'nn_raw': _bytes_feature(nn_raw),
        'ref_raw': _bytes_feature(ref_raw)}))
    out_file.write(example.SerializeToString())

    return 1

parser = argparse.ArgumentParser(description='convert wavs into record pairs')
parser.add_argument('--data', type=str, default='./data',
                    help='Directory containing datasets')
parser.add_argument('--tmp', type=str, default='./edited',
                    help='Directory containing pre-edited datasets')
parser.add_argument('--save_path', type=str, default='output/ksagan.tfrecords',
                    help='Path to save the dataset')
parser.add_argument('--skip_edit', action='store_true',
                    help='Skip editing')
parser.add_argument('--examples_per_vocab', '-e', type=int, default=50)

parser.add_argument('--annotation_file', type=str)
parser.add_argument('--add_negative_native', default=0.0, type=float)

opts = parser.parse_args()

annotation = set()
if opts.annotation_file:
    with open(opts.annotation_file) as f:
        for l in f:
            l = l.strip().split(" ")
            if l[2] == "True":
                annotation.add(l[1]+".wav")

s3re = re.compile(r's3_(\d{3})')

files = [f for f in listdir(opts.data) if isfile(join(opts.data, f))]

# apply noise reduction, and silence removal as file preparation

native_files=dict()
nonnative_files=dict()

for f in files:
    m = s3re.search(f)
    if not m:
        print('------- invalid file', f)
    else:
        filepath = join(opts.data, f)
        try:
            vocab = m.group(1)
            native = f[0] == 'K' or f[0] == 'k' or f in annotation
            if not opts.skip_edit:
                tfm = sox.Transformer()
                tfm.noiseprof(filepath, 'noise_profile')
                tfm.noisered('noise_profile', amount=0.05)
                tfm.silence(1,0.1)
                tfm.silence(-1,0.1)
                tfm.build(filepath, join(opts.tmp, f))
                duration = sox.file_info.duration(join(opts.tmp, f))
                print(native, vocab, f, sox.file_info.duration(filepath), duration)
            if native:
                if vocab not in native_files:
                    native_files[vocab]=[]
                native_files[vocab].append(f)
            else:
                if vocab not in nonnative_files:
                    nonnative_files[vocab]=[]
                nonnative_files[vocab].append(f)
        except Exception:
            print('ERROR: could not process>>>>', filepath)

# now build triplets: non-native, challenger native, reference native

out_file = tf.python_io.TFRecordWriter(opts.save_path)

count = 0
for i in range(opts.examples_per_vocab * len(nonnative_files)):
    k = random.choice(nonnative_files.keys())
    if random.random()<opts.add_negative_native:
        nk = k
        while nk == k:
            nk = random.choice(nonnative_files.keys())
        nonnative_example = random.choice(native_files[nk])
    else:
        nonnative_example = random.choice(nonnative_files[k])
    challenger_example = random.choice(native_files[k])
    print(nonnative_example, challenger_example)

    count += encoder_proc(join(opts.tmp,nonnative_example), join(opts.tmp,challenger_example), out_file, 2 ** 14)
print("--------------------------------- count=", count)