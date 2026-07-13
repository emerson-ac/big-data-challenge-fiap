"""Treina um modelo scikit-learn no dataset Iris e salva model.joblib.

Uso:
    python train.py            # gera ./model.joblib

O deploy faz o upload para s3://<bucket>/sklearn-iris/model.joblib,
que é o storageUri consumido pelo InferenceService.
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import joblib

X, y = datasets.load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, y)
joblib.dump(clf, "model.joblib")
print("Saved model.joblib")
