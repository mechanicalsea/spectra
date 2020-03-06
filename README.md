# spectra_torch

This library provides common spectra features from an audio signal including MFCCs and filter bank energies. This library mimics the library [`python_speech_features`](https://github.com/jameslyons/python_speech_features) but **PyTorch-style**.

This library provides voice activity detection (VAD) based on energy. This library mimics the library [`VAD-python`](https://github.com/marsbroshok/VAD-python) but **PyTorch-style**.

Use: Rui Wang. (2020, february 26). mechanicalsea/spectra: release v0.2.2 (Version 0.2.2).

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

Easy demo:

```python
import spectra_torch as mm
import torchaudio as ta

sig, sr = ta.load_wav('mywav.wav')
sig = sig[0]
mfcc = mm.mfcc(sig, sr) # MFCC
starts, detection = is_speech(sig, sr, speechlen=1) # VAD
```

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

## Reference

- `python_speeck_features`: https://github.com/jameslyons/python_speech_features
- `VAD-python`: https://github.com/marsbroshok/VAD-python
- `pythonaudio`: https://pytorch.org/audio/_modules/torchaudio/functional.html