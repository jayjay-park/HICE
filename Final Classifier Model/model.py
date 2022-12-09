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

web_user_embededings = pd.read_csv("web-redditEmbeddings-subreddits.csv", header=None)

embdeding_subs = set(list(web_user_embededings[0]))

for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" in name:
        name = name[10:]
    if ".csv" in name:
        name = name[: -4]

    if name.lower() not in embdeding_subs:
        embedding = np.array([float(0) for _ in range(300)])
    else:
        for r in web_user_embededings.iterrows():
            if r[1][0] == name.lower():
                l = list(r[1][1:])
                l = [float(x) for x in l]
                embedding = np.array([l])
    embedding = embedding.reshape(300,)
    # print(embedding)

    train_path = "./train/subreddit_" + name + ".csv"

    df = pd.read_csv(train_path)
    df = df[["controversiality", "body", "score"]]
    matrix = []
    for r in df.iterrows():
        # print(r[1][0], r[1][2])
        a = np.array([float(r[1][0])])
        b = np.array([float(r[1][2])])
        temp = np.concatenate([a, b])
        tensor = r[1][1][13: -34].split()
        tensor = np.array([float(x) for x in tensor])
        tensor = np.array([np.sum(tensor)])
        # print(tensor.shape)
        matrix.append(np.concatenate([temp, tensor]))
    matrix = np.array(matrix)
    # print(matrix.shape)
    if matrix.shape[0] < 7000:
        matrix = np.pad(matrix, ((0, 7000 - matrix.shape[0]), (0, 0)), "constant", constant_values=float(0))
    matrix = matrix.flatten()
    matrix = np.concatenate([matrix, embedding])
    # print(matrix.shape, matrix)
    all_data.append(matrix)

all_data = np.array(all_data)
# all_data = np.zeros(493, 21300)
write = pd.DataFrame(all_data)
write.to_csv("embeddings.csv")
print(all_data)
print(all_data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=10)

# Create model
print("Building model")
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile model
print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.AUC()])
model.fit(X_train, y_train, epochs = 10)
model.evaluate(X_test, y_test)
model.save("model", save_format='h5')