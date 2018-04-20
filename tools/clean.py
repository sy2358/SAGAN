import sox
import argparse

parser = argparse.ArgumentParser(description='clean wav')
parser.add_argument('--file', type=str)
parser.add_argument('--out', type=str)

opts = parser.parse_args()

tfm = sox.Transformer()
tfm.noiseprof(opts.file, 'noise_profile')
tfm.noisered('noise_profile', amount=0.05)
tfm.silence(1,0.1)
tfm.silence(-1,0.1)
tfm.build(opts.file, opts.out)
duration = sox.file_info.duration(opts.out)
print(sox.file_info.duration(opts.file), duration)

