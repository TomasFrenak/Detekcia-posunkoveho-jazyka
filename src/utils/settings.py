import os

# Language used in the app
LANGUAGE = 'SPJ'

# Version 1 or 2 of the app
VERSION = 2

# The hand user will sign with if version 1 of the app is used
HAND = 'RIGHT'  # 'RIGHT', 'LEFT' or 'BOTH'

# Number of records wanted for each gesture in V1 of the app
HAND_IMAGES = 10

# Number of records wanted for each gesture in V2 of the app
N_SEQUENCES = 10
SEQUENCE_LENGTH = 20

# Number of data points collected by MediaPipe holistic model
POINTS_ON_HAND = 21  # Points per hand
POINTS_ON_FACE = 468  # Points on face
POINTS_ON_POSE = 33  # Points on pose
DATA_POINTS_PER_HAND = POINTS_ON_HAND * 3  # 3 cords (x,y, z) per data point on hand
DATA_POINTS_ON_FACE = POINTS_ON_FACE * 3  # 3 cords (x,y, z) per data point on face
DATA_POINTS_ON_POSE = POINTS_ON_POSE * 3  # 3 cords (x,y, z) per data point on pose

# Calculate the number of data points
DATA_POINTS = DATA_POINTS_PER_HAND * 2 + DATA_POINTS_ON_FACE + DATA_POINTS_ON_POSE if (
        VERSION == 2) else (DATA_POINTS_PER_HAND if HAND != 'BOTH' else DATA_POINTS_PER_HAND * 2)

# PATHS
DATA_DIR = '../data'
DATA_DIR_V2 = DATA_DIR + f'/{LANGUAGE}' + '/v2'
DATA_PATH = DATA_DIR + f'/{LANGUAGE}' + '/data.csv'
MODEL_PATH = DATA_DIR + f'/{LANGUAGE}' + '/model.joblib' if VERSION == 1 else DATA_DIR + f'/{LANGUAGE}/' + '/modelV2.joblib'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)