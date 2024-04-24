import os
import mediapipe as mp

# The hand user will sign with
HAND = 'RIGHT'

# Number of records wanted for each gesture
HAND_IMAGES = 10

# Number of data points collected by MediaPipe holistic model
POINTS_ON_HAND = 21  # Points per hand
DATA_POINTS_PER_HAND = POINTS_ON_HAND * 2  # 2 cords (x,y) per data point on hand
DATA_POINTS = DATA_POINTS_PER_HAND if HAND != 'BOTH' else DATA_POINTS_PER_HAND * 2  # 1 or 2 hands

# PATHS
DATA_DIR = '../data'
DATA_PATH = DATA_DIR + '/data.csv'
MODEL_PATH = DATA_DIR + '/model.joblib'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
