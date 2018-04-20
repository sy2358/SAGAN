from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import SEGAN, CLASSIFY
import os
from tensorflow.python.client import device_lib
from scipy.io import wavfile
from data_loader import pre_emph
import regex as re

devices = device_lib.list_local_devices()

flags = tf.app.flags
flags.DEFINE_integer("seed",111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 150, "Epochs to train (Def: 150).")
flags.DEFINE_integer("batch_size", 150, "Batch size (Def: 150).")
flags.DEFINE_integer("save_freq", 50, "Batch save freq (Def: 50).")
flags.DEFINE_integer("canvas_size", 2**14, "Canvas size (Def: 2^14).")
flags.DEFINE_integer("denoise_epoch", 5, "Epoch where noise in disc is "
                                          "removed (Def: 5).")
flags.DEFINE_integer("l1_remove_epoch", 150, "Epoch where L1 in G is "
                                           "removed (Def: 150).")
flags.DEFINE_boolean("squeeze_generator", False,
                     "do not use generator in discriminator training (train classifier)")
flags.DEFINE_boolean("bias_deconv", False,
                     "Flag to specify if we bias deconvs (Def: False)")
flags.DEFINE_boolean("bias_downconv", False,
                     "flag to specify if we bias downconvs (def: false)")
flags.DEFINE_boolean("bias_D_conv", False,
                     "flag to specify if we bias D_convs (def: false)")
# TODO: noise decay is under check
flags.DEFINE_float("denoise_lbound", 0.01, "Min noise std to be still alive (Def: 0.001)")
flags.DEFINE_float("noise_decay", 0.7, "Decay rate of noise std (Def: 0.7)")
flags.DEFINE_float("d_label_smooth", 0.25, "Smooth factor in D (Def: 0.25)")
flags.DEFINE_float("init_noise_std", 0.5, "Init noise std (Def: 0.5)")
flags.DEFINE_float("init_l1_weight", 100., "Init L1 lambda (Def: 100)")
flags.DEFINE_integer("z_dim", 256, "Dimension of input noise to G (Def: 256).")
flags.DEFINE_integer("z_depth", 256, "Depth of input noise to G (Def: 256).")
flags.DEFINE_string("save_path", "segan_results", "Path to save out model "
                                                   "files. (Def: dwavegan_model"
                                                   ").")
flags.DEFINE_string("g_nl", "leaky", "Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
flags.DEFINE_string("model", "gan", "Type of model to train: gan or ae. (Def: gan).")
flags.DEFINE_string("deconv_type", "deconv", "Type of deconv method: deconv or "
                                             "nn_deconv (Def: deconv).")
flags.DEFINE_string("g_type", "ae", "Type of G to use: ae or dwave. (Def: ae).")
flags.DEFINE_float("g_learning_rate", 0.0002, "G learning_rate (Def: 0.0002)")
flags.DEFINE_float("d_learning_rate", 0.0002, "D learning_rate (Def: 0.0002)")
flags.DEFINE_float("beta_1", 0.5, "Adam beta 1 (Def: 0.5)")
flags.DEFINE_float("preemph", 0.95, "Pre-emph factor (Def: 0.95)")
flags.DEFINE_string("synthesis_path", "dwavegan_samples", "Path to save output"
                                                          " generated samples."
                                                          " (Def: dwavegan_sam"
                                                          "ples).")
flags.DEFINE_string("e2e_dataset", "data/segan.tfrecords", "TFRecords"
                                                          " (Def: data/"
                                                          "segan.tfrecords.")
flags.DEFINE_string("save_clean_path", "test_clean_results", "Path to save clean utts")
flags.DEFINE_string("test_wav", None, "name of test wav (it won't train)")
flags.DEFINE_string("ref_wav", None, "name of ref wav (it will classify test_wav through generator or not)")
flags.DEFINE_string("weights", None, "Weights file")
FLAGS = flags.FLAGS

def pre_emph_test(coeff, canvas_size):
    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def read_wave(sess, test_wav):
  fm, wav_data = wavfile.read(FLAGS.test_wav)
  wavname = FLAGS.test_wav.split('/')[-1]
  if fm != 16000:
      raise ValueError('16kHz required! Test file is different')
  wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
  if FLAGS.preemph  > 0:
      print('preemph test wave with {}'.format(FLAGS.preemph))
      x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
      wave = sess.run(preemph_op, feed_dict={x_pholder:wave})
  print('test wave shape: ', wave.shape)
  print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
  return wave


def read_signals(sess, wav_file, wav_canvas_size):
    fm, wav_data = wavfile.read(wav_file)

    max_size = len(wav_data)

    if max_size > wav_canvas_size:
      return None
    max_size=wav_canvas_size

    wav_data_pad = np.zeros(max_size, dtype=np.int32)
    wav_data_pad[:wav_data.shape[0]] = wav_data

    wave_norm = (2./65535.) * (wav_data_pad.astype(np.float32) - 32767) + 1.
    if FLAGS.preemph  > 0:
      x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave_norm.shape[0])
      wave_norm = sess.run(preemph_op, feed_dict={x_pholder:wave_norm})

    return wave_norm

s3re = re.compile(r's3_(\d{3})')

def main(_):
    print('Parsed arguments: ', FLAGS.__flags)

    # make save path if it is required
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    if not os.path.exists(FLAGS.synthesis_path):
        os.makedirs(FLAGS.synthesis_path)
    np.random.seed(FLAGS.seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    udevices = []
    for device in devices:
        if len(devices) > 1 and 'CPU' in device.name:
            # Use cpu only when we dont have gpus
            continue
        print('Using device: ', device.name)
        udevices.append(device.name)
    print("!!!!!!", udevices)
    # execute the session
    with tf.Session(config=config) as sess:
        if FLAGS.squeeze_generator:
            print('Creating Classifier')
            se_model = CLASSIFY(sess, FLAGS, udevices)
        elif FLAGS.model == 'gan':
            print('Creating GAN model')
            se_model = SEGAN(sess, FLAGS, udevices)
        elif FLAGS.model == 'ae':
            print('Creating AE model')
            se_model = SEAE(sess, FLAGS, udevices)
        else:
            raise ValueError('{} model type not understood!'.format(FLAGS.model))
        if FLAGS.test_wav is None:
            se_model.train(FLAGS, udevices)
        else:
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')
            print('Loading model weights...')
            se_model.load(FLAGS.save_path, FLAGS.weights)

            if FLAGS.ref_wav is None:
              wave = read_wave(sess, FLAGS.test_wav)
              c_wave = se_model.clean(wave)
              print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
              wavfile.write(os.path.join(FLAGS.save_clean_path, wavname), 16000, c_wave)
              print('Done cleaning {} and saved '
                    'to {}'.format(FLAGS.test_wav,
                                   os.path.join(FLAGS.save_clean_path, wavname)))
            else:
              files=[]
              if os.path.isdir(FLAGS.test_wav):
                for filename in os.listdir(FLAGS.test_wav):
                  if filename.endswith(".wav"):
                    files.append(os.path.join(FLAGS.test_wav, filename))
              else:
                files.append(FLAGS.test_wav)
              for f in files:
                refs=[]
                if os.path.isdir(FLAGS.ref_wav):
                  m = s3re.search(f)
                  if m:
                    vid = m.group(1)
                    for filename in os.listdir(FLAGS.ref_wav):
                      if filename.endswith(".wav") and filename.find("s3_"+vid) != -1:
                        refs.append(os.path.join(FLAGS.ref_wav, filename))
                else:
                  refs.append(FLAGS.ref_wav)
                score=[]
                wav_signals = read_signals(sess, f, FLAGS.canvas_size)
                if wav_signals is not None:
                  for r in refs:
                    ref_signals = read_signals(sess, r, FLAGS.canvas_size)
                    if ref_signals is not None:
                      logit = se_model.classify(wav_signals, ref_signals)
                      score.append(str(logit))
                print("RES\t", f, "\t".join(score))

if __name__ == '__main__':
    tf.app.run()
