import os
import pandas as pd
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
labels_csv = pd.read_csv("revised_labels.csv")

# Iterate through all csv files to pull nxm matrices
# Consideer each matrix like an image for binary classification
# Reshape each matrix to fit model input
for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" not in name:
        name = "subreddit_" + name
    path = "./raw/" + name + ".csv"
    if not os.path.isfile(path):
        continue
    df = pd.read_csv(path)
    df = df[["controversiality", "body", "score"]]
    temp = []
    for text in df["body"]:
        if type(text) != str:
            temp.append([0])
        else:
            temp.append(embed([text]))
    df["body"] = temp
    df.to_csv("./train/" + name + ".csv")