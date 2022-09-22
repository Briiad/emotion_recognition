import pandas as pd
import os
import sys

# # Library for machine learning
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split

# # Library for building CNN
# import tensorflow as tf
# import keras
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
# from keras.utils import to_categorical, np_utils

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# READING THE DATA
crema = "./input/speech-en/AudioWAV/"

crema_dir = os.listdir(crema)

file_emotion  = []
file_path = []

for file in crema_dir:
  file_path.append(crema + file)
  part = file.split('_')

  if part[2] == 'SAD':
    file_emotion.append('sad')
  elif part[2] == 'ANG':
    file_emotion.append('angry')
  elif part[2] == 'DIS':
    file_emotion.append('disgust')
  elif part[2] == 'FEA':
    file_emotion.append('fear')
  elif part[2] == 'HAP':
    file_emotion.append('happy')
  elif part[2] == 'NEU':
    file_emotion.append('neutral')
  else :
    file_emotion.append('None')

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

crema_df = pd.concat([emotion_df, path_df], axis=1)
crema_df.to_csv('crema_data.csv', index=False)
crema_df.head()

df = pd.concat([crema_df], axis = 0)
print(df.Emotions.value_counts())