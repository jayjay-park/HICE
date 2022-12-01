import pandas as pd
import numpy as np

labels_csv = pd.read_csv("revised_labels.csv")
web_user_embededings = pd.read_csv("web-redditEmbeddings-subreddits.csv", header=None)

embdeding_subs = set(list(web_user_embededings[0]))

for row in labels_csv["subreddit"]:
    name = row[2: -2]
    if "subreddit_" in name:
        name = name[10:]
    if ".csv" in name:
        name = name[: -4]
    # print(name)
    if name.lower() not in embdeding_subs:
        print(name.lower(), " is not in the list of embeddings")

for row in web_user_embededings.iterrows():
    print(list(row[1][1:]))