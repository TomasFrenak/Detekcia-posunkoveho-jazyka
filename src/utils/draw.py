import cv2
import mediapipe as mp

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def draw_hand_info(frame, prediction):
    # Draws the information about the hand and predicted gesture on the screen
    (h, w) = frame.shape[:2]
    (text_width, text_height), _ = cv2.getTextSize(prediction, FONT_FACE, 4,  6)

    cv2.putText(frame, f"{prediction}", ((w - text_width) // 2, h - 50), FONT_FACE, 4, (0, 0, 365), 6,
                cv2.LINE_AA)


def draw_capture_info(frame, hand_gesture, mode):
    # Draws the menu on the screen
    if mode == 0:
        text = f"Press 'R' to start capture '{hand_gesture}'"
        draw_text_rect(frame, text, 30, 20, (233, 196, 209))
        cv2.putText(frame, text, (30, 50), FONT_FACE, 1.3, (146, 27, 49), 3, cv2.LINE_AA)

    elif mode == 1:
        text = f"Press 'C' to record current position for '{hand_gesture}'"
        draw_text_rect(frame, text, 30, 20, (233, 196, 209))
        cv2.putText(frame, text, (30, 50), FONT_FACE, 1.3, (146, 27, 49), 3, cv2.LINE_AA)

    return frame


def draw_hand_landmarks(frame, rh, lh):
    mp.solutions.drawing_utils.draw_landmarks(frame, rh, mp.solutions.hands.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(frame, lh, mp.solutions.hands.HAND_CONNECTIONS)


def draw_text_rect(frame, text, x, y, color: tuple[int, int, int]):
    # Draws a rectangle around the text
    text_size = cv2.getTextSize(text, FONT_FACE, 1.3, 3)
    cv2.rectangle(frame, (x - 10, y - 10), (x + text_size[0][0] + 10, y + text_size[0][1] + 10), color, -1)


def draw_text(frame, text, x, y, rect_color=(233, 196, 209)):
    draw_text_rect(frame, text, x, y - 30, rect_color)
    cv2.putText(frame, text, (x, y), FONT_FACE, 1.3, (146, 27, 49), 3, cv2.LINE_AA)