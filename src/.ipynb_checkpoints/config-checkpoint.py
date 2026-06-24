# # =========================
# # 🧠 Parkinson EEG Project Config
# # =========================

# import os

# # -------------------------
# # PATHS
# # -------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA_DIR = os.path.join(BASE_DIR, "data")
# RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
# PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
# MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")

# # -------------------------
# # EEG SETTINGS
# # -------------------------
# SFREQ = 256

# BANDS = {
#     "delta": (1, 4),
#     "theta": (4, 8),
#     "alpha": (8, 12),
#     "beta": (12, 30)
# }

# # -------------------------
# # DATASET SETTINGS
# # -------------------------
# LABEL_MAP = {
#     "hc": 0,
#     "pd": 1
# }

# GENDER_MAP = {
#     "m": 0,
#     "f": 1
# }

# SEED = 42

# # -------------------------
# # TRAINING SETTINGS
# # -------------------------
# TEST_SIZE = 0.2
# RANDOM_STATE = 42

# MONTE_CARLO_RUNS = 20

# # -------------------------
# # MODELS CONFIG
# # -------------------------
# MODEL_PARAMS = {
#     "random_forest": {
#         "n_estimators": 200,
#         "max_depth": None,
#         "random_state": RANDOM_STATE
#     },

#     "svm": {
#         "C": 1.0,
#         "kernel": "rbf",
#         "probability": True
#     },

#     "logistic_regression": {
#         "max_iter": 1000
#     },

#     "knn": {
#         "n_neighbors": 5
#     }
# }

# # -------------------------
# # FEATURE ENGINEERING
# # -------------------------
# USE_SEASON_PROXY = True
# SEASON_MODE = "mod4"

# # -------------------------
# # VISUALIZATION
# # -------------------------
# FIGSIZE = (10, 6)
# DPI = 300
# STYLE = "seaborn-v0_8"

# # -------------------------
# # FILE NAMES
# # -------------------------
# PROCESSED_METADATA_FILE = "metadata.csv"



import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
EEG_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "eeg_features.csv")
FINAL_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "final_dataset.csv")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SFREQ = 256

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30)
}

LABEL_MAP = {
    "hc": 0,
    "pd": 1
}

SEED = 42
TEST_SIZE = 0.2
MONTE_CARLO_RUNS = 20