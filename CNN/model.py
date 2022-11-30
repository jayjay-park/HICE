import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

labels_csv = pd.read_csv("revised_labels.csv")
labels = list(labels_csv["labels"])
training_data = []

# create model
model = Sequential()
model.add(Dense(60, input_shape=(60,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['f1'])
# model.fit(training_data, training_labels, steps_per_epoch = 500, epochs = 10, callbacks = early_stopping)