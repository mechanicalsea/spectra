from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import time
import torch
import torchaudio as ta
import numpy as np
import matplotlib.pyplot as plt

# import spectra_torch package
import spectra_torch.base as mm

result = lambda x: print('%.6f - %.6f - %.6f' % (np.mean(np.abs(x)), np.max(np.abs(x)), np.min(np.abs(x))))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    wav_file = 'piece_20_32k.wav' # wav [-1, 1]
    rate, sig = wav.read(wav_file)
    signal, sr = ta.load(wav_file)
    signal = signal.to(device)
    signal = signal[0]

    # Demo: Voice Activity Detection
    detection, starts, signal = mm.is_speech(signal, samplerate=sr, winlen=0.02, hoplen=0.01,
                                             thresEnergy=0.6, speechlen=0.5, lowfreq=300,
                                             highfreq=3000, preemph=0.97)
    plt.figure(figsize=(12, 4))
    plt.plot(signal.cpu().numpy())
    plt.plot(starts.cpu().numpy(), detection.cpu().numpy()*0.1)
    plt.show()

    rate, sig = wav.read(wav_file)
    signal, sr = ta.load(wav_file)
    signal = signal.to(device)
    signal = signal[0]

    # Demo: vad MFCC
    detection, _, _ = mm.is_speech(signal, samplerate=sr, winlen=0.025, hoplen=0.01)
    tor_feat = mm.mfcc(signal, sr, winlen=0.025, hoplen=0.01)
    vad_feat = tor_feat[detection == 1.0]
    print('original mfcc: {}, vad mfcc: {}'.format(tor_feat.shape, vad_feat.shape))

    # Demo: MFCC
    mfcc_feat = mfcc(sig, rate, nfft=1024)
    tor_feat = mm.mfcc(signal, sr)
    df = mfcc_feat - tor_feat.cpu().numpy()

    d_mfcc_feat = delta(mfcc_feat, 2)
    t_d = mm.delta(tor_feat, 2)
    d1 = d_mfcc_feat - t_d.cpu().numpy()

    fbank_feat = logfbank(sig, rate, nfft=1024)
    t_log_fbank = mm.logfbank(signal, sr)
    fd = fbank_feat - t_log_fbank.cpu().numpy()

    result(df)
    result(d1)
    result(fd)


    print('Cost Evalution tested on Ubuntu 18.04: Wish Positive')
    n = 40
    # load by scipy
    s1 = time.time()
    for i in range(n):
        (rate, sig) = wav.read(wav_file)
    t1 = time.time()
    # load by torchaudio
    s2 = time.time()
    for i in range(n):
        signal, sr = ta.load(wav_file)  # load varies from load_wav
        signal = signal[0].to(device)
    t2 = time.time()
    # mfcc by python_speech_features
    s3 = time.time()
    for i in range(n):
        mfcc_feat = mfcc(sig, rate, nfft=1024)
    t3 = time.time()
    # mfcc by spectra_torch
    s4 = time.time()
    for i in range(n):
        tor_feat = mm.mfcc(signal, sr)
    t4 = time.time()

    print("Gap loading %.5f(scipy %.5f, torchaudio %.5f)" % (((t1-s1) - (t2-s2))/n, (t1-s1)/n, (t2-s2)/n))
    print("Gap mfcc %.5f(python_speech_features %.5f, spectra_torch %.5f)" % (((t3-s3) - (t4-s4))/n, (t3-s3)/n, (t4-s4)/n))
    print("Negative means the torch-style can run slower than the numpy-style.")
    print("Otherwise What a nice day!")
