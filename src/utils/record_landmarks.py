import csv

import click
import cv2
import numpy as np

import src.utils.settings as s

def get_gestures_to_record():
    with open(f'{s.DATA_DIR}/capture.txt', 'r') as file:
        gestures = file.read().splitlines()
    print(f'Gestures to record: {gestures}')

    return gestures


def check_if_gesture_exists(hand_gestures: list):
    with open(f'{s.DATA_DIR}/data.csv', encoding='utf-8-sig') as file:
        current_gesture_data_csv = csv.reader(file)
        current_gesture_data = list(current_gesture_data_csv)
        current_gestures = [row[0] for row in current_gesture_data][1:]
        existing_gestures = set([g for g in current_gestures if g in hand_gestures])

        if existing_gestures:
            print(f"'{', '.join(existing_gestures)}' already exists in the dataset")
            if click.confirm('Do you want to overwrite the data?', default=False):
                return hand_gestures, [row for row in current_gesture_data if row[0] not in hand_gestures]
            else:
                if click.confirm("Would you like to record more images for this gesture instead? ", default=False):
                    return hand_gestures, current_gesture_data
                else:
                    if len(existing_gestures) == len(hand_gestures):
                        raise SystemExit('All gestures already exist in the dataset')
                    else:
                        return list(set(hand_gestures) - existing_gestures), current_gesture_data

        else:
            return hand_gestures, current_gesture_data


def calc_hand_rect(frame, landmarks):
    frame_h, frame_w, _ = frame.shape

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        l_x = min(int(landmark.x * frame_w), frame_w - 1)
        l_y = min(int(landmark.y * frame_h), frame_h - 1)

        landmark_point = [np.array((l_x, l_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def mark_hand_position(multi_hand_landmarks):
    data_aux, x_, y_ = [], [], []

    for hand_landmarks in multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    return data_aux
