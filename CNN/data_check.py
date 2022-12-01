import pandas as pd
import os

labels_csv = pd.read_csv("revised_labels.csv")
l = []

for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" not in name:
        name = "subreddit_" + name
    if ".csv" not in name:
        train_path = "./train/" + name + ".csv"
        raw_path = "./raw/" + name + ".csv"
    else:
        train_path = "./train/" + name
        raw_path = "./raw/" + name

    # print(train_path)
    if not os.path.isfile(raw_path):
        print("doesn't exist: ", name)
        continue
    if not os.path.isfile(train_path):
        # l.append(name)
        print("missing: ", name)

# l.sort()
# print(l)