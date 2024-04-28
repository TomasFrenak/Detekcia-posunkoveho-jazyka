import os

import datetime
import joblib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from utils.settings import DATA_DIR, DATA_DIR_V2,DATA_POINTS, MODEL_PATH, SEQUENCE_LENGTH


gestures = np.array([os.path.splitext(filename)[0] for filename in os.listdir(DATA_DIR_V2)])

label_map = {label: i for i, label in enumerate(gestures)}

data, labels = [], []
for gesture in gestures:
    gesture_data = np.load(f'{DATA_DIR_V2}/{gesture}.npy')
    for sequence in gesture_data:
        data.append(sequence)
        labels.append(label_map[gesture])

X, y = np.array(data), to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

tb_callback = TensorBoard(log_dir=DATA_DIR+f'/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH,DATA_POINTS)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(gestures.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion matrix: {metrics.multilabel_confusion_matrix(test_labels, predictions)}')

joblib.dump(model, MODEL_PATH)
