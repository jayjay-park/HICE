# Input:
# - Comment file (several cols: controversiality, body, score)
# - Community - Community embedding
# - Community - User embedding
# Idea: flatten matrices and concat them to feed to NN
# Idea: feed a vector of 3 matrices (tensor-ish)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

labels_csv = pd.read_csv("revised_labels.csv")
labels = list(labels_csv["labels"])
labels = [[x] for x in labels]
labels = np.array(labels)
all_data = []

for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" not in name:
        name = "subreddit_" + name
    if ".csv" not in name:
        train_path = "./train/" + name + ".csv"
    else:
        train_path = "./train/" + name

    df = pd.read_csv(train_path)
    df = df[["controversiality", "body", "score"]].to_numpy()
    if len(df) < 7000:
        for i in range(len(df), 7000):
            df = np.append(df, np.array[0, tf.zeros([1, 512], dtype=tf.float32), 0], axis = 0)
    all_data.append(df)

all_data = np.array(all_data)
print(all_data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=10)

# Create model
print("Building model")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile model
print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 10)
model.evaluate(X_test, y_test)