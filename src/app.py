import cv2
import os
import numpy as np

from utils.draw import *
from utils.helpers import *
import utils.settings as s


def collect_data(gestures_to_record: list, current_gesture_data: list):
    running, recording = True, False
    gestures_recorded, record_index, mode = 0, 0, 0
    new_gesture_data = []

    holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

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


def collect_data_v2(gestures_to_record: list):

    holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    for gesture in gestures_to_record:
        gesture_data = []
        for sequence in range(s.N_SEQUENCES):
            sequence_data = []
            for frame_n in range(s.SEQUENCE_LENGTH):
                if frame_n == 0:
                    waiting_time = 50

                    while waiting_time > 0:
                        _, frame = cap.read()
                        frame = cv2.flip(frame, 1)
                        draw_text(frame, f"Prepare to record {gesture}. Sequence number: {sequence}",30,50, rect_color=(114, 128, 250))

                        cv2.imshow('Hand Gesture Recorder', frame)
                        cv2.waitKey(1)
                        waiting_time -= 1
                else:
                    _, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    draw_text(frame, f"Recording data for {gesture}. Sequence number: {sequence}",30,50, rect_color= (175, 225, 175))

                frame = cv2.flip(frame, 1)

                result = process_frame(frame, holistic_model)
                rh, lh = result.right_hand_landmarks, result.left_hand_landmarks
                draw_hand_landmarks(frame, rh, lh)

                frame = cv2.flip(frame, 1)

                cv2.imshow('Hand Gesture Recorder', frame)
                cv2.waitKey(1)

                data = get_all_gesture_data(result)
                sequence_data.append(data)
            gesture_data.append(sequence_data)
        save_data = np.array(gesture_data)
        np.save(f'{s.DATA_DIR_V2}/{gesture}.npy', save_data)


def recognize_gestures():
    running = True

    holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
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
            draw_hand_info(frame, prediction=prediction[0])

        cv2.imshow('Hand Gesture Recognition', frame)


def recognize_gestures_v2():
    running = True
    sequence = []
    word = ''
    frame_count = 0
    gestures = np.array([os.path.splitext(filename)[0] for filename in os.listdir(s.DATA_DIR_V2)])

    holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
    prediction_model = joblib.load(s.MODEL_PATH)

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

        data = get_all_gesture_data(results)

        sequence.append(data)

        if rh or lh:
            sequence = sequence[-s.SEQUENCE_LENGTH:]
            if len(sequence) == s.SEQUENCE_LENGTH:
                prediction = prediction_model.predict(np.expand_dims(sequence, axis=0))
                sequence = []

                if np.amax(prediction) > 0.75:
                    word = gestures[np.argmax(prediction)]
                    print(f"{gestures[np.argmax(prediction)]} - {np.max(prediction)}")

        if word:
            draw_hand_info(frame, prediction=word)
            frame_count += 1
            if frame_count == s.SEQUENCE_LENGTH:
                frame_count = 0
                word = ''

        cv2.imshow('Hand Gesture Recognition', frame)


def main():
    if s.VERSION == 1:
        gestures_to_record = get_gestures_to_record()

        if gestures_to_record:
            gestures_to_record, current_gesture_data = check_if_gesture_exists(gestures_to_record)

            if gestures_to_record:
                collect_data(gestures_to_record, current_gesture_data)
                clear_capture()
        else:
            recognize_gestures()

    elif s.VERSION == 2:
        gestures_to_record = get_gestures_to_record()

        if gestures_to_record:
            collect_data_v2(gestures_to_record)
            clear_capture()
        else:
            recognize_gestures_v2()


if __name__ == '__main__':
    main()
