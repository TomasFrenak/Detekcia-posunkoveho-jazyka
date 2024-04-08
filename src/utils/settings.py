import os
import mediapipe as mp

# Number of records wanted for each gesture
HAND_IMAGES = 5

# PATHS
DATA_DIR = '../data'
DATA_PATH = DATA_DIR + '/data.csv'
MODEL_PATH = DATA_DIR + '/model.joblib'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Hand recognition initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.7,
    max_num_hands=2
)


