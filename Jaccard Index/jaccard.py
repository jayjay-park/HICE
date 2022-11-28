import pandas as pd
import csv

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# Create set of hate keywords
df = pd.read_csv("Enlarged_hate_keywords.csv", header=None)
hate_keywords = set()
# print(df)

for i, row in df.iterrows():
    # print(row[0])
    hate_keywords.add(row[0])

# print(hate_keywords)


# Create dictionary of topics for each subreddit
df2 = pd.read_csv("top_words_entire_subreddit_hatred.csv", header=None)
subreddit_topics = {}
# print(df2)

for i, row in df2.iterrows():
    # print(row[1], row[2])
    if row[1] not in subreddit_topics:
        subreddit_topics[row[1]] = {row[2]}
    else:
        subreddit_topics[row[1]].add(row[2])

# print(subreddit_topics["['tipofmytongue']"])



# Calculate Jaccard for each subreddit
jaccard_indices = {}
for k, v in subreddit_topics.items():
    jaccard_indices[k] = jaccard(list(v), list(hate_keywords))

with open("jaccard_indices_hatred.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    for row in jaccard_indices.items():
        writer.writerow(row)