# You can extract speech feature via torch-style code.
# There are some available operation to be reused, such as rfft, dct, delta
# Thank you for project "python_speech_features"
# Thank you for project "VAD-python"
# Author: Rui Wang 2020
import torch
import torchaudio.functional as AF
import math
import logging


# Reference: https://github.com/jameslyons/python_speech_features
# The style of this module mimics the `python_speech_features`
# Reference: https://github.com/marsbroshok/VAD-python
# The style of this module mimics the `VAD-python`
# A details can be found in the reference.

# Note that the precision of torch is float32 default. So there is a bit of gap between torch and numpy.


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
    framelen = round(framelen)  # round_half_up(framelen)
    framehop = round(framehop)  # round_half_up(framehop)
    if slen <= framelen:
        nframes = 1
    else:
        nframes = 1 + int(math.ceil((
                                                1.0 * slen - framelen) / framehop))  # 1 + int(torch.ceil(torch.tensor((1.0 * slen - framelen) / framehop)))

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
        lift = 1 + (ceplifter / 2.) * torch.sin(math.pi * torch.arange(numcep) / ceplifter)
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


def stereo2mono(signal):
    """
    Convert stereo to mono.
    :param signal: (nchannels, time)
    :return: (time)
    """
    if signal.shape[0] == 2 and signal.ndim == 2:
        return torch.mean(signal, dim=1)
    elif signal.shape[0] == 1 and signal.ndim == 2:
        return signal[0]
    elif signal.ndim == 1:
        return signal


def calculate_frequencies(frame, samplerate):
    """
    Calculate half frequencies (0, sample rate) within a frame.
    :param frame: (framelen,)
    :param samplerate:
    :return: (framelen//2,) or or (framelen//2 - 1,)
    """
    # 半频
    n = len(frame)
    t = frame.shape[0] * 1.0 / samplerate
    if samplerate % 2 == 0:
        return torch.arange(1, n // 2 + 1) / t
    else:
        return torch.arange(1, (n - 1) // 2 + 1) / t


def calculate_energy(frame):
    """
    Calculate energy of a frame by rfft.
    :param frame: (framelen,)
    :return: (framelen//2,) or (framelen//2 - 1,) that equals to half frequencies
    """
    mag = torch.norm(torch.rfft(frame, 1), dim=1)[1:]
    energy = mag.T ** 2
    return energy


def freq_energy(frame, samplerate):
    """
    Calculate a pair of (frequencies, energy) of a frame.
    :param frame: (framelen,)
    :param samplerate:
    :return: both (framelen//2,) or (framelen//2-1,)
    """
    freq = calculate_frequencies(frame, samplerate)
    energy = calculate_energy(frame)
    return freq, energy


def energy_ratio(frame, samplerate, lowfreq, highfreq):
    """
    Calculate the ratio between energy of speech band and total energy for window.
    :param frame: (framelen,)
    :param samplerate:
    :param lowfreq:
    :param highfreq:
    :return: float
    """
    freq, energy = freq_energy(frame, samplerate)
    voice_energy = torch.sum(energy[(freq > lowfreq) & (freq < highfreq)])
    full_energy = torch.sum(energy)
    if full_energy == 0: full_energy = 2.220446049250313e-16  # 极小正数替换0
    return voice_energy / full_energy


def smooth_detection(detect, winlen, speechlen):
    """
    Apply median filter with length of {speechlen} to smooth detected speech regions
    :param detect: (nframes,)
    :param winlen:
    :param speechlen:
    :return: (nframes,)
    """
    medianwin = max(int(speechlen / winlen), 1)
    if medianwin % 2 == 0:
        medianwin -= 1
    mid = (medianwin - 1) // 2
    y = torch.zeros((len(detect), medianwin), dtype=detect.dtype)
    y[:, mid] = detect
    for i in range(mid):
        j = mid - i
        y[j:, i] = detect[:-j]
        y[:j, i] = detect[0]
        y[:-j, -(i + 1)] = detect[j:]
        y[-j:, -(i + 1)] = detect[-1]
    medianEnergy = torch.median(y.type(torch.float), dim=1).values.type(torch.bool)
    return medianEnergy


def is_speech(signal, samplerate=16000, winlen=0.02, hoplen=0.01, thresEnergy=0.6, speechlen=0.5,
              lowfreq=300, highfreq=3000, preemph=0.97):
    """
    Use signal energy to detect voice activity in PyTorch's Tensor.
    Detects speech regions based on ratio between speech band energy and total energy.
    Outputs are two tensor with the number of frames where the first output is start frame
        and the second output is to indicate voice activity.
    :param signal: (time,)
    :param samplerate:
    :param winlen:
    :param hoplen:
    :param thresEnergy:
    :param speechlen:
    :param lowfreq:
    :param highfreq:
    :param preemph:
    :return: (nframes,), (nframes)
    """
    if len(signal) < round(winlen * samplerate):
        return torch.tensor([0]), torch.tensor([0])
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, hoplen * samplerate)
    starts = torch.arange(0, frames.shape[0]) * round(hoplen * samplerate)
    detection = torch.zeros(frames.shape[0], dtype=torch.bool)
    for i, frame in enumerate(frames):
        ratioEnergy = energy_ratio(frame, samplerate, lowfreq, highfreq)
        detection[i] = ratioEnergy > thresEnergy
    if speechlen:
        detection = smooth_detection(detection, winlen, speechlen)
    return starts, detection

