import pandas as pd

from config import (
    METADATA_PATH,
    EEG_FEATURES_PATH,
    FINAL_DATASET_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

eeg = pd.read_csv(
    EEG_FEATURES_PATH
)

eeg["participant_id"] = (
    eeg["file"]
    .str.split("_")
    .str[0]
)

merged = pd.merge(
    metadata,
    eeg,
    on="participant_id",
    how="inner"
)

merged.to_csv(
    FINAL_DATASET_PATH,
    index=False
)

print(
    "Merged dataset saved:"
)

print(
    FINAL_DATASET_PATH
)

print(
    merged.shape
)