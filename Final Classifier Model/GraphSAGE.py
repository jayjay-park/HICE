import pandas as pd

truth = pd.read_csv("revised_labels.csv")
subreddits = list(truth["subreddit"])
t = {}
for i in range(len(subreddits)):
    temp = subreddits[i]
    name = temp[2:-2]
    if "subreddit_" in name:
        name = name[10:]
    if ".csv" in name:
        name = name[: -4]
    t[name.lower()] = subreddits[i]
subreddits = t

new_src_tar = pd.DataFrame(columns=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"])
aggregated_labels = {}

col = [i for i in range(301)]
new_com_use = pd.DataFrame(columns=col)

df_com_com = pd.read_csv('soc-redditHyperlinks-title (1).tsv', sep='\t')
com_com_src_names = set(list(df_com_com["SOURCE_SUBREDDIT"]))

df_com_user = pd.read_csv('web-redditEmbeddings-subreddits.csv', header=None)
embdeding_subs = set(list(df_com_user[0]))

for index, row in truth.iterrows():
    temp = row[1]
    name = temp[2:-2]
    if "subreddit_" in name:
        name = name[10:]
    if ".csv" in name:
        name = name[: -4]
    # print(name)
    if name.lower() in com_com_src_names:
        temp_df = df_com_com.loc[df_com_com["SOURCE_SUBREDDIT"] == name.lower()][["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"]]
        new_src_tar = new_src_tar.append(temp_df, ignore_index=True)
        if name.lower() not in embdeding_subs:
            zero_features = [name.lower()] + [float(0) for _ in range(300)]
            zero_features = pd.DataFrame(zero_features).T
            new_com_use = new_com_use.append(zero_features, ignore_index=True)
            aggregated_labels[name.lower()] = row[3]
        else:
            new_com_use = new_com_use.append(df_com_user.loc[df_com_user[0] == name.lower()], ignore_index=True)
            aggregated_labels[name.lower()] = row[3]
print(new_src_tar)
print(new_com_use)
com_com_tgt_names = set(list(new_src_tar["TARGET_SUBREDDIT"]))
tar_diff =  com_com_tgt_names - set(subreddits)
for n in tar_diff:
    # print(n)
    # print(new_src_tar.index[new_src_tar["TARGET_SUBREDDIT"] == n].tolist())
    new_src_tar = new_src_tar.drop(new_src_tar.index[new_src_tar["TARGET_SUBREDDIT"] == n.lower()])
new_src_tar.to_csv("aggregated_data.csv")

src_tar_names = set(list(new_src_tar["SOURCE_SUBREDDIT"]) + list(new_src_tar["TARGET_SUBREDDIT"]))
print(len(src_tar_names))
new_embdeding_subs = set((list(new_com_use[0])))
for d in src_tar_names:
    # print(truth.loc[truth["subreddit"] == subreddits[d.lower()]]["labels"])
    if d.lower() not in new_embdeding_subs:
        if d.lower() not in embdeding_subs:
            zero_features = [d.lower()] + [float(0) for _ in range(300)]
            zero_features = pd.DataFrame(zero_features).T
            new_com_use = new_com_use.append(zero_features, ignore_index=True)
            # print(truth.loc[truth["subreddit"] == subreddits[d.lower()]]["labels"].iloc[0])
            aggregated_labels[d.lower()] = truth.loc[truth["subreddit"] == subreddits[d.lower()]]["labels"].iloc[0]
            # print(truth.loc[truth["subreddit"] == d.lower()])
        else:
            new_com_use = new_com_use.append(df_com_user.loc[df_com_user[0] == d.lower()], ignore_index=True)
            aggregated_labels[d.lower()] = truth.loc[truth["subreddit"] == subreddits[d.lower()]]["labels"].iloc[0]
            # print(truth.loc[truth["subreddit"] == subreddits[d.lower()]]["labels"].iloc[0])
            # print(truth.loc[truth["subreddit"] == d.lower()])

new_embdeding_subs = set((list(new_com_use[0])))
diff = new_embdeding_subs - src_tar_names
print(len(new_embdeding_subs), len(diff))
for d in diff:
    new_com_use = new_com_use.drop(new_com_use.index[new_com_use[0] == d])
    del aggregated_labels[d]

print(len(set((list(new_com_use[0])))))

new_com_use.to_csv("aggregated_embeddings.csv")
aggregated_labels = pd.Series(aggregated_labels)
aggregated_labels.to_csv("aggregated_labels.csv")

# com_com_names = set(list(df_com_com["SOURCE_SUBREDDIT"]) + list(df_com_com["TARGET_SUBREDDIT"]))
# # print(com_com_names)
# com_user_names = set(list(df_com_user[0]))
# # print(com_user_names)
# diff = com_com_names - com_user_names
# print(len(diff))
# for index, row in df_com_com.iterrows():
#     print(index)
#     if row["SOURCE_SUBREDDIT"] in diff or row["TARGET_SUBREDDIT"] in diff:
#         df_com_com.drop(index, inplace=True)

# df_com_com.to_csv("aggregated_data.csv")

# df = pd.read_csv("aggregated_data.csv")
# df_names = set(list(df["SOURCE_SUBREDDIT"]) + list(df["TARGET_SUBREDDIT"]))
# diff = com_user_names - df_names
# for index, row in df_com_user.iterrows():
#     print(index)
#     if row[0] in diff:
#         df_com_user.drop(index, inplace=True)

# df_com_user.to_csv("aggregated_embeddings.csv")