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
    df = df[["controversiality", "body", "score"]]
    matrix = []
    for r in df.iterrows():
        # print(r[1][0], r[1][2])
        a = np.array([r[1][0]])
        b = np.array([r[1][2]])
        temp = np.concatenate([a, b])
        tensor = r[1][1][13: -34].split()
        tensor = np.array([float(x) for x in tensor])
        # print(tensor.shape)
        matrix.append(np.concatenate([temp, tensor]))
    matrix = np.array(matrix)
    # print(matrix.shape)
    if matrix.shape[0] < 7000:
        matrix = np.pad(matrix, ((0, 7000 - matrix.shape[0]), (0, 0)), "constant")
    all_data.append(matrix)

all_data = np.array(all_data)
# a = np.array([0])
# b = np.array([1])
# temp = np.concatenate([a, b])
# tensor = np.array(tf.zeros([1, 512], dtype=tf.float32)).reshape(512,)
# print(temp.shape, tensor.shape)
# all_data = np.array([[np.concatenate([temp, tensor]) for _ in range(7000)] for x in range(493)])
print(all_data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=10)
# print(X_train, y_train)

# Create model
print("Building model")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile model
print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training")
model.fit(X_train, y_train, epochs = 10)
print("Done training")
model.evaluate(X_test, y_test)
model.save("model")