import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics

labels_csv = pd.read_csv("revised_labels.csv")
labels = list(labels_csv["labels"])
# labels = [[x] for x in labels]
labels = np.array(labels)
# print(labels)

final_embeddings = pd.read_csv("final_embeddings.csv")
all_data = np.array(final_embeddings)
# print(all_data)

X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=10)

final_rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 4, random_state = 913)
final_rf_model.fit(X_train, y_train.ravel())
predictions = final_rf_model.predict(X_test)
print(classification_report(y_test, predictions))
print(metrics.roc_auc_score(y_test, predictions))

final_svm = SVC(kernel="linear", C=10.0)
# print(X_train, y_train)
final_svm.fit(X_train, y_train.ravel())
preds = final_svm.predict(X_test)
print(classification_report(y_test, preds))
print(metrics.roc_auc_score(y_test, preds))