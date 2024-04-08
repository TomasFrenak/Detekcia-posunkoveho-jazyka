import csv
import joblib

import cv2
import numpy as np

from utils.record_landmarks import get_gestures_to_record, check_if_gesture_exists, mark_hand_position
from utils.draw import draw_hand_info, draw_menu
from utils.controls import select_mode
import utils.settings as s


def build_window(**kwargs):
    running = True
    mode = 0
    record = kwargs.get('record')

    if record:
        hands_recorded, record_index = 0, 0
        gestures_to_record = kwargs.get('gestures')
        current_gesture_data = kwargs.get('current_gesture_data')
        new_gesture_data = []

    model = joblib.load(s.MODEL_PATH)

    cap = cv2.VideoCapture(0)

    while running:
        key = cv2.waitKey(10)
        if key == 27:
            break

        mode, recording = select_mode(key, mode)

        check, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = s.hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                hand_data = mark_hand_position(results.multi_hand_landmarks)
                hand_rect = calc_hand_rect(frame, hand_landmarks)

                if record:
                    draw_hand_info(frame_copy, hand_rect, handedness=handedness)
                    if recording and hands_recorded < s.HAND_IMAGES:
                        hands_recorded += 1
                        hand_data.insert(0, gestures_to_record[record_index])
                        new_gesture_data.append(hand_data)
                        if hands_recorded == s.HAND_IMAGES:
                            hands_recorded = 0
                            record_index += 1
                            if record_index == len(gestures_to_record):
                                running = False
                    recording = False

                else:
                    prediction = model.predict([hand_data])
                    draw_hand_info(frame_copy, hand_rect, handedness=handedness, prediction=prediction)

                s.mp_drawing.draw_landmarks(
                    frame_copy,
                    hand_landmarks,
                    s.mp_hands.HAND_CONNECTIONS,
                    s.mp_drawing_styles.get_default_hand_landmarks_style(),
                    s.mp_drawing_styles.get_default_hand_connections_style()
                )

        if record:
            try:
                frame_copy = draw_menu(frame_copy, gestures_to_record[record_index], mode)
            except IndexError:
                break
        cv2.imshow('Hand Gesture Recognition', frame_copy)

    if record and new_gesture_data:
        print(current_gesture_data)
        print(new_gesture_data)
        current_gesture_data.extend(new_gesture_data)
        with open(f'{s.DATA_DIR}/data.csv', mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerows(current_gesture_data)

        print(f"Data saved to {s.DATA_DIR}/data.csv")
        print(f"In order to train the model on the new data, run 'python train.py'")


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


def main():

    gestures_to_record = get_gestures_to_record()
    if gestures_to_record:
        gestures_to_record, current_gesture_data = check_if_gesture_exists(gestures_to_record)
        build_window(record=True, gestures=gestures_to_record, current_gesture_data=current_gesture_data)

    else:
        build_window(record=False)


if __name__ == '__main__':
    main()