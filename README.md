# spectra_torch

> Considering the [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi) is presented, so it is more practical to use it.
> Also, [SpeechBrain](https://speechbrain.github.io/index.html), A PyTorch-based Speech Toolkit, is coming. I am looking forward to a nice step on speech.
> To conclude, this package is used to learn spectra of a signal, so it is valuable at all.

**News**: Tutorials continue to come! [Jupiter Notebook Viewer](https://nbviewer.jupyter.org/) for "Reaload?"er.

- 2020.03.22: The bandpass filter is [here](https://github.com/mechanicalsea/spectra/blob/master/notebooks/PyTorch%20Filter.ipynb).
- 2020.03.29: The parameterized bandpass filter is uploaded as "[Parameter Filter.ipynb](https://github.com/mechanicalsea/spectra/blob/master/notebooks/Parameterized%20Filter.ipynb)". Also, [core.py](https://github.com/mechanicalsea/spectra/blob/master/spectra_torch/core.py) add the new feature.

This library provides common spectra features from an audio signal including MFCCs and filter bank energies. This library mimics the library [`python_speech_features`](https://github.com/jameslyons/python_speech_features) but **PyTorch-style**.

This library provides voice activity detection (VAD) based on energy. This library mimics the library [`VAD-python`](https://github.com/marsbroshok/VAD-python) but **PyTorch-style**.

Use: Rui Wang. (2020, March 14). mechanicalsea/spectra: release v0.4.0 (Version 0.4.0).

## Installation

This library is avaliable on pypi.org

To install from Pypi:

```
pip install --upgrade spectra-torch
```

Require:

- python: 3.7.3
- torch: 1.4.0
- torchaudio: 0.4.0

## Usage

Supported features:

- Mel Frequency Cepstral Coefficients (MFCC)
- Filterbank Energies
- Log Filterbank Energies
- Voice Activity Detection (VAD)

Here are [examples](https://github.com/mechanicalsea/spectra/blob/master/examples.py).

### Easy demo:

```python
# Ensure cuda is available.
import spectra_torch.base as mm
import torchaudio as ta

sig, sr = ta.load_wav('piece_20_32k.wav')
sig = sig[0].cuda()
mfcc = mm.mfcc(sig, sr) # MFCC
starts, detection = mm.is_speech(sig, sr, speechlen=0.5) # VAD
```

### Tutorial

Tutorials of MFCC and VAD is provided at [notebooks](https://github.com/mechanicalsea/spectra/tree/master/notebooks).

Step-by-step description is presented. Welcome to enjoy it.

## Performance

The difference between `spectra_torch` and `python_speech_features`:

- Precision bais: 1e-4
- Speed up: 0.1s/mfcc

## MFCC

```python
def mfcc(signal, samplerate=16000, winlen=0.025, hoplen=0.01, 
         numcep=13, nfilt=26, nfft=None, lowfreq=0, highfreq=None, 
         preemph=0.97, ceplifter=22, plusEnergy=True)
```

## Filterbank

```python
def fbank(signal, samplerate=16000, winlen=0.025, hoplen=0.01, 
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
```

## VAD

```python
def is_speech(signal, samplerate=16000, winlen=0.02, hoplen=0.01, 
              thresEnergy=0.6, speechlen=0.5, lowfreq=300, highfreq=3000, 
              preemph=0.97)
```

## Parameterized Bandpass Filter

```python
class PFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, device="cpu",
                 mode='bandpass',sample_rate=16000, min_hz=50, max_hz=None,
                 min_band_hz=50, win_fn="Hamming")
```

## Reference

- `python_speeck_features`: https://github.com/jameslyons/python_speech_features
- `VAD-python`: https://github.com/marsbroshok/VAD-python
- `pythonaudio`: https://pytorch.org/audio/_modules/torchaudio/functional.html

Thanks for you attention.

Free for question to my email (rwang@tongji.edu.cn).