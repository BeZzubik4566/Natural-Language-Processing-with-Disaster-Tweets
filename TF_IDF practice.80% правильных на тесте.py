import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import skorch
from kagglesdk.competitions.types import submission_status
from narwhals import DataFrame
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from catboost import CatBoostClassifier
from scipy.sparse import hstack
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import f1_score



# ============================================================
# 1 шаг. ЗАГРУЖАЕМ ДАННЫЕ
# ============================================================

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train=df_train.drop('target',axis=1)
y_train=df_train['target']

# ============================================================
# 2 шаг. ПОДГОТАВЛИВАЕМ ТЕКСТ
# ============================================================

train_text = X_train['text'].fillna("")
test_text = df_test['text'].fillna("")


# ============================================================
# 3 шаг. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=200000,
    sublinear_tf=True
)

print("\nBuilding TF-IDF...")

X = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# ============================================================
# 4 шаг. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    C=2.0,
    solver="lbfgs",
    max_iter=300
)

print("\nTraining model...")

model.fit(X, y_train)
ypred = model.predict(X)

print("Training finished.")
print(f1_score(y_train, ypred))


# ============================================================
# 5 шаг. ПРЕДСКАЗЫВАЕМ TEST
# ============================================================

test_predict = model.predict(X_test)
#test_proba = model.predict(X_test)

# ============================================================
# 6 шаг. СОЗДАЁМ SUBMISSION
# ============================================================


submission = pd.DataFrame({
    "id": df_test["id"],
    "target": test_predict
})

submission.to_csv("submission.csv", index=False)