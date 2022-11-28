import pandas as pd
import csv
from collections import Counter

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# Create set of hate keywords
df = pd.read_csv("shorter_hate_keyword.csv", header=None)
hate_keywords = set()
# print(df)

for i, row in df.iterrows():
    # print(row[0])
    hate_keywords.add(row[0])

# print(hate_keywords)



# Create set of target keywords
# df = pd.read_csv("Target_cat_keywords.csv")
# target_keywords = set()

# for i, row in df.iterrows():
#     # print(row[2])
#     target_keywords.add(row[2])


# Create dictionary of topics for each subreddit
df2 = pd.read_csv("top_words_entire_subreddit_hatred.csv", header=None)
subreddit_topics = {}
count = 0 # count the number of topic words for each subreddit
# print(df2)

for i, row in df2.iterrows():
    # print(row[1], row[2])
    if row[1] not in subreddit_topics:
        subreddit_topics[row[1]] = ({row[2]},[row[2]])
    else:
        subreddit_topics[row[1]][0].add(row[2])
        subreddit_topics[row[1]][1].append(row[2])
    

print(subreddit_topics["['tipofmytongue']"])



# Calculate Jaccard for each subreddit
jaccard_indices = {}
for k, v in subreddit_topics.items():
    # jaccard_indices[k] = jaccard(list(v[0]), list(hate_keywords))
    jaccard_score = jaccard(list(v[0]), list(hate_keywords))
    weight = 0
    for key, value in Counter(v[1]).items():
        # print(key, value)
        if key in hate_keywords:
            weight += value

    jaccard_indices[k] = weight * jaccard_score

# print(jaccard_indices)

with open("revised_jaccard_indices_weighted_hatred.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    for row in jaccard_indices.items():
        writer.writerow(row)