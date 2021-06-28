import tensorflow as tf
#from tensorflow import keras
#python3 data_collect.py
#. data_update.sh

dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
data_dir = tf.keras.utils.get_file('cats-dogs.zip', origin=dataset_url, untar=True)
