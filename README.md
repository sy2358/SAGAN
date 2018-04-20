## SAGAN: Speech Assessment using Generative Adversarial Network

### Introduction

This project is based on the [SEGAN project](https://arxiv.org/abs/1703.09452) - and the code based on [CMUSphinx fork](https://github.com/cmusphinx/segan) adding tensorflow 1.4 support.

### Dependencies

* Python 2.7
* TensorFlow 1.4.1
* sox 1.3.3

You can install the requirements either to your virtualenv or the system via pip with:

```
pip install -r requirements.txt
```

### Data

The data is a collection of student and reference utterances of 300 Korean words produced by 50 native and 123 non-native speakers corpus L2 Korean Speech Corpus (L2KSC). Files starting with 'K' are gold version (Korean), other files are student recording.

Wav files (16Khz) are trimmed at the maximum using `sox` and only utterances below ~1s are kept for the training (to stay 2^14 input network size). Script `tools/clean.py` is denoising slightly and trimming initial/final silence.

The script `tools/prep_data.py` is then building random pairs of student/reference utterance for the same words and serialize data in TF record.

```
usage: prep_data.py [-h] [--data DATA] [--tmp TMP] [--save_path SAVE_PATH]
                    [--skip_edit] [--examples_per_vocab EXAMPLES_PER_VOCAB]
                    [--annotation_file ANNOTATION_FILE]
                    [--add_negative_native ADD_NEGATIVE_NATIVE]

convert wavs into record pairs

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Directory containing datasets
  --tmp TMP             Directory containing pre-edited datasets
  --save_path SAVE_PATH
                        Path to save the dataset
  --skip_edit           Skip editing
  --examples_per_vocab EXAMPLES_PER_VOCAB, -e EXAMPLES_PER_VOCAB
  --annotation_file ANNOTATION_FILE
  --add_negative_native ADD_NEGATIVE_NATIVE
```

By default, the script is performing cleanup (from `clean.py`) and saving edited wavs into tmp directory. To use pre-edited files, use `--skip_edit`.

By default, the script is generating random pairs of student/reference utterance for a given word. The option `--add_negative_native` (disabled by default) is sampling additional pairs of wrong/reference utterance pairs where the wrong utterance is part of the reference but for another word.

### Training

Once you have the TFRecords file created in `output/ksagan.tfrecords` you can simply run the training process with:

```
CUDA_VISIBLE_DEVICES="0" python2 main.py --init_noise_std 0. --save_path output/ksagan-withloss2 \
                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
                                          --save_freq 50 --epoch 86 --bias_deconv True \
                                          --e2e_dataset output/ksagan.tfrecords \
                                          --bias_downconv True --bias_D_conv True
```


### Loading model and classification



### Loading model and prediction



### Authors

* **Seung Hee Yang** (SNU)

### Contact

e-mail: sy2358@snu.ac.kr
