import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# model = tf.keras.models.load_model("model")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

test_data = pd.read_csv("./raw/subreddit_FatNiggerHat.csv")
test_data = test_data[["controversiality", "body", "score"]]

for row in test_data.iterrows():
    a = np.array([row[1][0]])
    b = np.array([row[1][2]])
    temp = np.concatenate([a, b])
    tensor = embed([row[1][1]]).numpy()
    print(tensor)