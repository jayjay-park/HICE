import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

labels_csv = pd.read_csv("revised_labels.csv")
labels = list(labels_csv["labels"])
labels = [[x] for x in labels]
labels = np.array(labels)

old_data = pd.read_csv("embeddings.csv")
old_data = old_data.drop(columns=["Unnamed: 0"])
final_cols = list(old_data.columns)

new_data = pd.read_csv("graph_embeddings.csv")
graph_subreddits = set(list(new_data["Unnamed: 0"]))
cols = list(new_data.columns)
# print(cols[1:3])

new_cols = pd.DataFrame()

for row in labels_csv["subreddit"]:
    name = row[2:-2]
    if "subreddit_" in name:
        name = name[10:]
    if ".csv" in name:
        name = name[: -4]

    if name in graph_subreddits:
        temp = new_data.loc[new_data["Unnamed: 0"] == name][["0", "1"]]
        # print(temp.columns)
        new_cols = pd.concat([new_cols, temp], axis=0, ignore_index=True)
    else:
        temp = pd.DataFrame([[float(0), float(0)]], columns=cols[1:3])
        # print(temp)
        new_cols = pd.concat([new_cols, temp], axis=0, ignore_index=True)

final_embeddings = old_data.copy(deep=True)
final_embeddings["21300"] = new_cols["0"]
final_embeddings["21301"] = new_cols["1"]
final_embeddings.to_csv("final_embeddings.csv")

all_data = np.array(final_embeddings)

X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=10)

evals = {}

# Create model
print("Building model")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile model
print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC()])
model.fit(X_train, y_train, epochs = 20)
evals["model1"] = model.evaluate(X_test, y_test)
model.save("model_v2.h5", save_format='h5')

# 1. k-fold cross validation
# 2. FalseNegative
# 3. epochs = 20
# 4. batch size specify -> small
# 5. change optimizer -> stochastic gradient descent

# Create model 2
print("Building model")
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(30, activation='relu'))
model2.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile model
print("Compiling model")
model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC()])
model2.fit(X_train, y_train, epochs = 20)
evals["model2"] = model2.evaluate(X_test, y_test)
model2.save("model_v2_stochastic.h5", save_format='h5')

# Create model 3
print("Building model")
fold1 = 1
evals["model3"] = []
kf = KFold(n_splits = 5, shuffle=True)
for train, test in kf.split(all_data, labels):
    model3 = tf.keras.Sequential()
    model3.add(tf.keras.layers.Dense(30, activation='relu'))
    model3.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    print("Compiling model")
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC()])
    checkpoint1 = tf.keras.callbacks.ModelCheckpoint("model_v2_kfold_" + str(fold1) + ".h5", monitor=tf.keras.metrics.AUC(), save_best_only=True, mode="max")
    model3.fit(all_data[train], labels[train], epochs = 20, callbacks=[checkpoint1])
    model3.save("model_v2_kfold_" + str(fold1) + ".h5", save_format='h5')
    model3.load_weights("model_v2_kfold_" + str(fold1) + ".h5")
    evals["model3"].append(model3.evaluate(all_data[test], labels[test]))
    tf.keras.backend.clear_session()
    fold1 += 1

# Create model 4
print("Building model")
fold2 = 1
evals["model4"] = []
kf2 = KFold(n_splits = 5, shuffle=True)
for train, test in kf2.split(all_data, labels):
    model4 = tf.keras.Sequential()
    model4.add(tf.keras.layers.Dense(30, activation='relu'))
    model4.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    print("Compiling model")
    model4.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC()])
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint("model_v2_stochastic_kfold_" + str(fold2) + ".h5", monitor=tf.keras.metrics.AUC(), save_best_only=True, mode="max")
    model4.fit(all_data[train], labels[train], epochs = 20, callbacks=[checkpoint2])
    model4.save("model_v2_stochastic_kfold_" + str(fold2) + ".h5", save_format='h5')
    model4.load_weights("model_v2_stochastic_kfold_" + str(fold2) + ".h5")
    evals["model4"].append(model4.evaluate(all_data[test], labels[test]))
    tf.keras.backend.clear_session()
    fold2 += 1


print(evals)