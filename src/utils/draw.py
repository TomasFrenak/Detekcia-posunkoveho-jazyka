import cv2


def draw_menu(frame, hand_gesture, mode):
    # Draws the menu on the screen
    if mode == 0:
        text = f"Press 'R' to start capture '{hand_gesture}'"
        draw_text_rect(frame, text, 30, 20, (233, 196, 209))
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (146, 27, 49), 3, cv2.LINE_AA)

    elif mode == 1 or mode == 2:
        text = f"Press 'C' to record current position for '{hand_gesture}'"
        draw_text_rect(frame, text, 30, 20, (233, 196, 209))
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (146, 27, 49), 3, cv2.LINE_AA)

    return frame


def draw_text_rect(frame, text, x, y, color: tuple[int, int, int]):
    # Draws a rectangle around the text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    cv2.rectangle(frame, (x - 10, y - 10), (x + text_size[0][0] + 10, y + text_size[0][1] + 10), color, -1)


def draw_hand_info(frame, hand_rect, **kwargs):
    # Draws the hand rectangle and the information about the hand and predicted gesture on the screen
    if hand_rect:
        cv2.rectangle(frame, (hand_rect[0] - 25, hand_rect[1] - 25), (hand_rect[2] + 25, hand_rect[3] + 25), (0, 0, 0),
                      3)

        if kwargs.get('handedness'):
            handedness = kwargs.get('handedness')

            hand = handedness.classification[0].label[0:]
            cv2.putText(frame, hand, (hand_rect[0], hand_rect[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        if kwargs.get('prediction'):
            prediction = kwargs.get('prediction')
            cv2.putText(frame, f": {prediction[0]}", (hand_rect[0] + 100, hand_rect[1] - 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)


# TODO: Implement this function or remove it
def draw_text(frame, text, x, y):
    draw_text_rect(frame, text, x, y + 20, (233, 196, 209))
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (146, 27, 49), 3, cv2.LINE_AA)
