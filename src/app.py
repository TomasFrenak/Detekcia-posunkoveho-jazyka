import cv2

from utils.draw import *
from utils.helpers import *
import utils.settings as s


def collect_data(gestures_to_record: list, current_gesture_data: list):
    running, recording = True, False
    gestures_recorded, record_index, mode = 0, 0, 0
    new_gesture_data = []

    holistic_model = s.holistic

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    while running:
        key = cv2.waitKey(10)
        if key == 27:
            break

        mode, recording = select_mode(key, mode)

        _, frame = cap.read()

        result = process_frame(frame, holistic_model)
        rh, lh = result.right_hand_landmarks, result.left_hand_landmarks
        draw_hand_landmarks(frame, rh, lh)

        frame = cv2.flip(frame, 1)
        frame = draw_capture_info(frame, gestures_to_record[record_index], mode)

        if recording and gestures_recorded < s.HAND_IMAGES:
            hand_data = get_hand_data(rh, lh).tolist()
            hand_data.insert(0, gestures_to_record[record_index])
            new_gesture_data.append(hand_data)
            gestures_recorded += 1

            if gestures_recorded == s.HAND_IMAGES:
                gestures_recorded = 0
                record_index += 1
                if record_index == len(gestures_to_record):
                    running = False

        cv2.imshow('Hand Gesture Recorder', frame)

    if new_gesture_data:
        save_data(new_gesture_data, current_gesture_data)


def recognize_gestures():
    running = True

    holistic_model = s.holistic
    prediction_model = load_saved_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    while running:
        key = cv2.waitKey(10)
        if key == 27:
            break

        _, frame = cap.read()

        results = process_frame(frame, holistic_model)
        rh, lh = results.right_hand_landmarks, results.left_hand_landmarks
        draw_hand_landmarks(frame, rh, lh)

        frame = cv2.flip(frame, 1)

        hand_data = get_hand_data(rh, lh)

        prediction_proba = prediction_model.predict_proba([hand_data])
        probabilities = prediction_proba[0]
        confidence = max(probabilities)
        prediction = prediction_model.classes_[probabilities.argmax()]

        if confidence > 0.5:
            print(f"{confidence} -> {prediction}")

        draw_hand_info(frame, prediction=prediction)

        cv2.imshow('Hand Gesture Recognition', frame)


def main():
    gestures_to_record = get_gestures_to_record()

    if gestures_to_record:
        gestures_to_record, current_gesture_data = check_if_gesture_exists(
            gestures_to_record)

        if gestures_to_record:
            collect_data(gestures_to_record, current_gesture_data)
    else:
        recognize_gestures()


if __name__ == '__main__':
    main()
