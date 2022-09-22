import matplotlib.pyplot as plt
import librosa
import numpy as np

from IPython.display import Audio
from main import df

# Spectogram and waveplots
def create_waveplot(data, sr, e):
  plt.figure(figsize=(10,3))
  plt.title('Waveplot audio with {} emotion' .format(e), size=15)
  librosa.display.waveshow(data, sr=sr)
  plt.show()

def create_spectrogram(data, sr, e):
  X = librosa.stft(data)
  Xdb = librosa.amplitude_to_db(abs(X))
  plt.figure(figsize=(12,3))
  plt.title('Spectogram audio with {} emotion' .format(e), size=15)
  librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
  plt.colorbar()
  plt.show()

# EMOTIONS
emotion='fear' # change this to see different emotions
path = np.array(df.Path[df.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path, autoplay=True, rate=sampling_rate)
