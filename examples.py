from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import time
import torchaudio as ta
import numpy as np
import matplotlib.pyplot as plt

# import spectra_torch package
import spectra_torch.base as mm

result = lambda x: print('%.6f - %.6f - %.6f' % (np.mean(np.abs(x)), np.max(np.abs(x)), np.min(np.abs(x))))

if __name__ == "__main__":
    wav_file = 'singing-01-003.wav'
    # Demo: Voice Activity Detection
    signal, sr = ta.load_wav(wav_file)
    signal = signal[0]
    starts, detection = mm.is_speech(signal, sr, winlen=0.02, hoplen=0.01, thresEnergy=0.6, speechlen=1)
    plt.figure(1)
    plt.plot(signal.numpy()/signal.abs().max().numpy())
    plt.plot(starts.numpy(), detection.numpy())
    plt.show()

    # Demo: MFCC
    (rate,sig) = wav.read(wav_file)
    signal, sr = ta.load_wav(wav_file) # load varies from load_wav
    signal = signal[0]

    mfcc_feat = mfcc(sig, rate)
    tor_feat = mm.mfcc(signal, sr)
    df = mfcc_feat - tor_feat.numpy()

    d_mfcc_feat = delta(mfcc_feat, 2)
    t_d = mm.delta(tor_feat, 2)
    d1 = d_mfcc_feat - t_d.numpy()

    fbank_feat = logfbank(sig,rate)
    t_log_fbank = mm.logfbank(signal, sr)
    fd = fbank_feat - t_log_fbank.numpy()

    result(df)
    result(d1)
    result(fd)


    print('Cost Evalution tested on MacOS 15.1: Wish negative')
    n = 40
    s1 = time.time()
    for i in range(n):
        (rate, sig) = wav.read('data/singing-01-001.wav')
    t1 = time.time()

    s2 = time.time()
    for i in range(n):
        signal, sr = ta.load_wav('data/singing-01-001.wav')  # load varies from load_wav
        signal = signal[0]
    t2 = time.time()

    s3 = time.time()
    for i in range(n):
        mfcc_feat = mfcc(sig, rate)
    t3 = time.time()

    s4 = time.time()
    for i in range(n):
        tor_feat = mm.mfcc(signal, sr)
    t4 = time.time()

    print("Gap loading %.3f(%.3f, %.3f)" % (((t1-s1) - (t2-s2))/n, (t1-s1)/n, (t2-s2)/n))
    print("Gap mfcc %.3f(%.3f, %.3f)" % (((t3-s3) - (t4-s4))/n, (t3-s3)/n, (t4-s4)/n))
    print("Negative means the torch-style can run faster than the numpy-style.")
    print("Otherwise How a pity!")
