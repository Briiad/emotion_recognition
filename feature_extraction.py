import numpy as np
import librosa
import pandas as pd

from main import df

# DATA AUGMENTATION
def noise(data):
  noise_amp = 0.035*np.random.uniform()*np.amax(data)
  data = data + noise_amp * np.random.normal(size=data.shape[0])
  return data

def shift(data):
  s_range = int(np.random.uniform(low=-5, high=5)*1000)
  return np.roll(data, s_range)

def stretch(data, rate=0.8):
  data = librosa.effects.time_stretch(data, rate)
  return data

def pitch(data, sampling_rate, pitch_factor=0.7):
  data = librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
  return data

path = np.array(df.Path)[1]
data, sampling_rate = librosa.load(path)

# FEATURE EXTRACTION
def extract_features(data):
  result = np.array([])
  # MFCC
  mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate).T, axis=0)
  result = np.hstack((result, mfcc))

  return result

def get_features(path):
  result = []

  # without augment
  res1 = extract_features(data)
  result = np.array(res1)

  # with noise
  noise_data = noise(data)
  res2 = extract_features(noise_data)
  result = np.vstack((result, res2))

  # with stretch and pitch
  stretch_data = stretch(data)
  pitch_data = pitch(stretch_data, sampling_rate)
  res3 = extract_features(pitch_data)
  result = np.vstack((result, res3))

  return result

X, Y = [], []
for path, emotion in zip(df.Path, df.Emotions):
  feature = get_features(path)
  for element in feature:
    X.append(element)
    Y.append(emotion)
    print(element, emotion)

print(len(X), len(Y), df.Path.shape)

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
Features.head()
