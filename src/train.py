import joblib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import utils.settings as s


X_dataset = np.loadtxt(s.DATA_PATH, delimiter=',', dtype='float32', usecols=list(range(1, 42 + 1)))
y_dataset = np.loadtxt(s.DATA_PATH, delimiter=',', dtype='str', usecols=0, encoding='utf-8-sig')

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly !'.format(score * 100))

joblib.dump(model, f'{s.MODEL_PATH}')
