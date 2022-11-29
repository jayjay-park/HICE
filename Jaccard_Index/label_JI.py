import pandas as pd
import csv

#prev_labels = pd.read_csv(".\Jaccard_Index\labels_Copy.csv", "r", 
#                           lineterminator='\n', header=None, sep=",", delimiter=None)

prev_label = pd.read_csv("shorter_weighted_JI_entire.csv", names=["subreddit", "jaccard_score"])

# print(prev_label)

# 1. should update labels_Copy's JI score to revised one
labels = []
for i, r in prev_label.iterrows():
    if r[1] > 0.17:
        labels.append(1)
    else:
        labels.append(0)

# 2. check labels based on the threshold 0.17
prev_label["labels"] = labels
# print(prev_label)
# 3. final groundtruth generated -> 494 + ? = 
prev_label.to_csv("revised_labels.csv", sep="\t")
# annotate -> label 1 -> kappa score -> if we have decent score

# then, feed forward NN -> to automatically label rest of the dataset