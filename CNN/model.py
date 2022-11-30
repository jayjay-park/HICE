import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

labels_csv = pd.read_csv("revised_labels.csv")
labels = list(labels_csv["labels"])
training_data = []

# Iterate through all csv files to pull nxm matrices
for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" not in name:
        name = "subreddit_" + name
    path = "./" + name + ".csv"
    df = pd.read_csv(path)
    print(df)
# Consideer each matrix like an image for binary classification
# Reshape each matrix to fit model input

# create model
model = Sequential()
model.add(Dense(60, input_shape=(60,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['f1'])
# model.fit(training_data, training_labels, steps_per_epoch = 500, epochs = 10, callbacks = early_stopping)