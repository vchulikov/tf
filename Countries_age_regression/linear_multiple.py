import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#link:
#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

# WSA - world system analysys
column_names = ['Cigars per capita', 'GDP per capita', 'Age', 'WSA country type', 'Country']

raw_dataset = pd.read_csv("smoking_data.csv", names=column_names,
                          na_values='?', comment='\t',
                          sep=',', skipinitialspace=True)

#remove column
del raw_dataset['Country']

dataset = raw_dataset.copy()
#clear dataset
dataset = dataset.dropna()

#modify dataset, create new columns corresponds to country type by WSA classification
dataset['WSA country type'] = dataset['WSA country type'].map({1: 'Central', 2: 'SemiPer', 3: 'Per'})
dataset = pd.get_dummies(dataset, columns=['WSA country type'], prefix='', prefix_sep='')
dataset.tail()

#divide all dataset to train and test samples
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#make set of features (determine the age by GDP)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Age')
test_labels = test_features.pop('Age')

#normalize dataset. One makes the fit more stable.
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

#show the normalized values of the first example
#first = np.array(train_features[:1])
#with np.printoptions(precision=2, suppress=True):
#  print('First example:', first)
#  print('Normalized:', normalizer(first).numpy())

#model

linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])

#about model
#linear_model.summary()
#linear_model.layers[1].kernel

linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

history = linear_model.fit(
    train_features, train_labels, 
    epochs=1000,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.01)

#plot_loss(history)
#plt.show()

#save learning results
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


#show learning results
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
#plt.show()

print("Linear model")
print(linear_model.evaluate(test_features, test_labels, verbose=0))

