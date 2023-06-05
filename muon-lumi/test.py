import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

column_names = ['Mu', 'M2R1', 'M2R2', 'M2R3', 'M2R4']

raw_dataset = pd.read_csv("test.csv", names=column_names,
                          na_values='?', comment='\t',
                          sep=',', skipinitialspace=True)


train_dataset = raw_dataset.sample(frac=0.9, random_state=2)
test_dataset  = raw_dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Mu')
test_labels = test_features.pop('Mu')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))


#model
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.01))
  return model
  
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(train_features, train_labels, validation_split=0.1, verbose=0, epochs=500)


#make predictions
print(test_features)
test_predictions = dnn_model.predict(test_features).flatten()

#draw predictions
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Mu]')
plt.ylabel('Predictions [Mu]')
lims = [0, 11]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()

def make_predictions(dnn, np_arr):
    data_f = {'M2R1': [np_arr[0]],
              'M2R2': [np_arr[1]],
              'M2R3': [np_arr[2]],
              'M2R4': [np_arr[3]]}
              
    df = pd.DataFrame(data_f)
    my_prediction = dnn.predict(df).flatten()

    print(my_prediction)



numpy_array = np.array([0.68, 0.68, 0.18, 0.03])

make_predictions(dnn_model, numpy_array)
