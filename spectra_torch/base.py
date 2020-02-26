# You can extract speech feature via torch-style code.
# There are some available operation to be reused, such as rfft, dct, delta
# Thank you for project "python_speech_features"
# Author: Rui Wang 2020
import torch
import torchaudio.functional as AF
import decimal
import math
import logging

pi = math.pi


# Reference: https://github.com/jameslyons/python_speech_features/
# The style of this module mimics the `python_speech_features`
# A details can be found in the reference.

# Note that the precision of torch is float32 default. So there is a bit of gap between torch and numpy.


# def round_half_up(number):
#     return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def preemphasis(signal, preemph=0.97):
    """
    Pre-emphasis on the input signal
    :param signal: (time,)
    :param preemph:
    :return: (time,)
    """
    return torch.cat((signal[0:1], signal[1:] - preemph * signal[:-1]))


def framesig(signal, framelen, framehop, winfunc=lambda x: torch.ones((x,))):
    """
    Frame a signal into overlapping frames.
    :param signal: (time,)
    :param framelen:
    :param framehop:
    :param winfunc:
    :return: (nframes, framelen)
    """
    slen = len(signal)
    framelen = round(framelen)#round_half_up(framelen)
    framehop = round(framehop)#round_half_up(framehop)
    if slen <= framelen:
        nframes = 1
    else:
        nframes = 1 + int(math.ceil((1.0 * slen - framelen) / framehop))#1 + int(torch.ceil(torch.tensor((1.0 * slen - framelen) / framehop)))

    padlen = int((nframes - 1) * framehop + framelen)

    zeros = torch.zeros((padlen - slen,))
    padsignal = torch.cat((signal, zeros))

    indices = torch.arange(0, framelen).view((1, -1)) \
              + torch.arange(0, nframes * framehop, framehop).view((-1, 1))
    frames = padsignal[indices]
    win = winfunc(framelen).view((1, -1))
    return frames * win


def magspec(frames, nfft):
    """
    Compute the magnitude spectrum of each frame.
    :param frames: (nframes, framelen)
    :param nfft:
    :return: (nframes, nfft//2+1)
    """
    # torch.rfft 取窗口大小为 nfft，计算结果的实部与虚部分开
    if frames.shape[1] > nfft:
        logging.warning(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            frames.shape[1], nfft)
        frames = frames[:, :nfft]
    else:
        frames = torch.cat((frames, torch.zeros((frames.shape[0], nfft - frames.shape[1]))), 1)
    complex_spec = torch.rfft(frames, 1)
    return torch.norm(complex_spec, dim=2)


def powspec(frames, nfft):
    """
    Compute the power spectrum of each fram.
    :param frames: (nframes, framelen)
    :param nfft:
    :return: (nframes, nfft//2+1)
    """
    return 1.0 / nfft * (magspec(frames, nfft) ** 2)


def calculate_nfft(samplerate, winlen):
    """
    Calculate the nfft which is a power of 2 and is larger than or euqal to winlen.
    :param samplerate:
    :param winlen:
    :return: int
    """
    winsize = winlen * samplerate
    nfft = 1
    while nfft < winsize:
        nfft <<= 1
    return nfft


def mfcc(signal, samplerate=16000, winlen=0.025, hoplen=0.01,
         numcep=13, nfilt=26, nfft=None, lowfreq=0, highfreq=None,
         preemph=0.97, ceplifter=22, plusEnergy=True,
         winfunc=lambda x: torch.ones((x,))):
    """
    Compute MFCC from an audio signal.
    :param signal: (time,)
    :param samplerate:
    :param winlen:
    :param hoplen:
    :param numcep: The number of cepstrum to retain.
    :param nfilt:
    :param nfft:
    :param lowfreq:
    :param highfreq:
    :param preemph:
    :param ceplifter:
    :param plusEnergy:
    :param winfunc:
    :return: (nframes, numcep)
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal, samplerate, winlen, hoplen, nfilt,
                         nfft, lowfreq, highfreq, preemph, winfunc)
    feat = torch.log(feat)
    feat = feat.mm(AF.create_dct(numcep, nfilt, norm='ortho'))
    feat = lifter(feat, ceplifter)
    if plusEnergy: feat[:, 0] = torch.log(energy)
    return feat


def fbank(signal, samplerate=16000, winlen=0.025, hoplen=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None,
          preemph=0.97, winfunc=lambda x: torch.ones((x,))):
    """
    Compute Mel-filterbank energy features from an audio signal.
    :param signal: (time,)
    :param samplerate:
    :param winlen:
    :param hoplen:
    :param nfilt: The number of filters in the filterbank.
    :param nfft:
    :param lowfreq:
    :param highfreq:
    :param preemph:
    :param winfunc:
    :return: (nframes, nfilt), (frames,)
    """
    highfreq = highfreq or samplerate >> 1
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, hoplen * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = torch.sum(pspec, 1)
    energy[energy == 0] = 2.220446049250313e-16  # 极小正数替换0

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = torch.mm(pspec, fb.T)
    feat[feat == 0] = 2.220446049250313e-16  # 极小正数替换0

    return feat, energy  # feat: size (winsize, nfilt)


def logfbank(signal, samplerate=16000, winlen=0.025, hoplen=0.01,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None,
             preemph=0.97, winfunc=lambda x: torch.ones((x,))):
    """
    Compute log Mel-filterbank energy features from an audio signal as log(fbank).
    :param signal: (time,)
    :param samplerate:
    :param winlen:
    :param hoplen:
    :param nfilt: The number of filters in the filterbank.
    :param nfft:
    :param lowfreq:
    :param highfreq:
    :param preemph:
    :param winfunc:
    :return: (nframes, nfilt)
    """
    feat, _ = fbank(signal, samplerate, winlen, hoplen, nfilt,
                         nfft, lowfreq, highfreq, preemph, winfunc)
    return torch.log(feat)


def hz2mel(hz):
    """
    Convert a value in Hertz(Hz) to Mels
    :param hz: float
    :return: float
    """
    return 2595 * math.log10(1 + hz / 700.)


def mel2hz(mel):
    """
    Convert a value in Mels to Hertz(Hz).
    :param mel: float
    :return: float
    """
    return 700 * (10 ** (mel / 2595.) - 1)


def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Create a matrix of Mel-filterbank.
    :param nfilt:
    :param nfft:
    :param samplerate:
    :param lowfreq:
    :param highfreq:
    :return: (nfilt, nfft//2+1)
    """
    highfreq = highfreq or samplerate >> 1
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = torch.linspace(lowmel, highmel, nfilt + 2)
    bin = torch.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = torch.zeros((nfilt, (nfft >> 1) + 1))
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank  # size (nfilt, nfft//2+1)


def lifter(cepstra, ceplifter=22):
    """
    Apply a cepstral lifer to the matrix of cepstra.
    :param cepstra: (nframes, numcep)
    :param ceplifter:
    :return: (nframes, numcep)
    """
    if ceplifter > 0:
        nframes, numcep = cepstra.shape
        lift = 1 + (ceplifter / 2.) * torch.sin(pi * torch.arange(numcep) / ceplifter)
        return lift * cepstra
    else:
        return cepstra


def delta(specgram, N):
    """
    Compute delta features from a feature vector sequence.
    :param specgram: (nframes, fealen), fealen is generally numcep in the MFCC.
    :param N:
    :return: (nframes, fealen)
    """
    # specgram: size (freq, time)
    return AF.compute_deltas(specgram.T.unsqueeze(0), (N << 1) + 1).squeeze(0).T
