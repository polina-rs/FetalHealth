import pandas as pd
import sklearn as sk
import sklearn.model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data = pd.read_csv('fetal_health.csv')
X = data.iloc[:, range(len(data.columns) - 1)]
y = data["fetal_health"]
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

# random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)  # predicts fetal health results based on medical parameters from X

y_pred = clf.predict(X_test)
print((pd.Series(y_pred).to_frame()).describe())
print(y_test.describe())
res = f1_score(y_test, y_pred, average='weighted')

print("f1 score: ", res)

is_same = (y_pred == y_test)
accuracy = is_same.sum()/len(is_same)

print(f"accuracy: {str(accuracy)}")