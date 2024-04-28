import click
import csv
import cv2
import joblib
import numpy as np

from src.utils.settings import DATA_DIR, DATA_PATH, MODEL_PATH, HAND, DATA_POINTS_PER_HAND, DATA_POINTS_ON_FACE, DATA_POINTS_ON_POSE


def check_if_gesture_exists(hand_gestures: list):
    try:
        with open(DATA_PATH, encoding='utf-8-sig') as file:
            current_gesture_data_csv = csv.reader(file)
            current_gesture_data = list(current_gesture_data_csv)
            current_gestures = [row[0] for row in current_gesture_data][1:]
            existing_gestures = set(g for g in current_gestures if g in hand_gestures)

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

    except FileNotFoundError:
        return hand_gestures, []


def get_gestures_to_record():
    try:
        with open(f'{DATA_DIR}/capture.txt', 'r') as file:
            gestures = file.read().splitlines()
        print(f'Gestures to record: {gestures}')
    except FileNotFoundError:
        gestures = []
        file = open(f'{DATA_DIR}/capture.txt', 'w')
        file.close()
        print(f'Creating file: {DATA_DIR}/capture.txt \nOnly recognition will be carried out')

    return gestures


def get_hand_data(rh, lh):
    if HAND == 'RIGHT':
        landmarks = np.array([[pos.x, pos.y, pos.z] for pos in rh.landmark]).flatten() if rh else np.zeros(DATA_POINTS_PER_HAND)
    elif HAND == 'LEFT':
        landmarks = np.array([[pos.x, pos.y, pos.z] for pos in lh.landmark]).flatten() if lh else np.zeros(DATA_POINTS_PER_HAND)
    else:
        rh_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in rh.landmark]).flatten() if rh else np.zeros(DATA_POINTS_PER_HAND)
        lh_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in lh.landmark]).flatten() if lh else np.zeros(DATA_POINTS_PER_HAND)

        landmarks = np.concatenate([rh_landmarks, lh_landmarks])

    return landmarks

def get_all_gesture_data(results):
    rh, lh = results.right_hand_landmarks, results.left_hand_landmarks
    face, pose = results.face_landmarks, results.pose_landmarks

    rh_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in rh.landmark]).flatten() if rh else np.zeros(DATA_POINTS_PER_HAND)
    lh_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in lh.landmark]).flatten() if lh else np.zeros(DATA_POINTS_PER_HAND)

    face_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in face.landmark]).flatten() if face else np.zeros(DATA_POINTS_ON_FACE)
    pose_landmarks = np.array([[pos.x, pos.y, pos.z] for pos in pose.landmark]).flatten() if pose else np.zeros(DATA_POINTS_ON_POSE)

    return np.concatenate([rh_landmarks, lh_landmarks, face_landmarks, pose_landmarks])

def load_saved_model():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise SystemExit('ERROR: Model not found. Please run the program in record mode and train the model by running'
                         ' "python train.py" first')

    return model


def process_frame(frame, holistic_model):
    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(rgb_frame)
    frame.flags.writeable = True

    return results


def save_data(new_data: list, current_data: list):
    current_data.extend(new_data)

    with open(DATA_PATH, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(current_data)

    print(f"Data successfully saved at {DATA_PATH}")


def select_mode(key, mode):
    if key == ord('r'):
        return 1, False
    elif key == ord('c'):
        return mode, True
    else:
        return mode, False


def clear_capture():
    with open(f'{DATA_DIR}/capture.txt', 'w') as file:
        file.write('')
    print("'capture.txt' cleared")