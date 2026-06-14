import os
import numpy as np
import pandas as pd
import mne

from config import (
    RAW_DATA_DIR,
    EEG_FEATURES_PATH
)


# -------------------------
# EEG PREPROCESSING
# -------------------------
def preprocess(file_path):

    raw = mne.io.read_raw_bdf(
        file_path,
        preload=True,
        verbose=False
    )

    raw.filter(
        l_freq=1,
        h_freq=40,
        verbose=False
    )

    raw.notch_filter(
        freqs=50,
        verbose=False
    )

    raw.set_eeg_reference(
        "average",
        verbose=False
    )

    return raw


# -------------------------
# BAND POWER
# -------------------------
def bandpower(data, sf, band):

    low, high = band

    freqs = np.fft.rfftfreq(
        len(data),
        d=1 / sf
    )

    psd = np.abs(
        np.fft.rfft(data)
    ) ** 2

    idx = (
        (freqs >= low)
        &
        (freqs <= high)
    )

    return np.mean(psd[idx])


# -------------------------
# FEATURE EXTRACTION
# -------------------------
# def extract_features(raw):

#     data = raw.get_data()

#     sf = raw.info["sfreq"]

#     channel_features = []

#     for channel in data:

#         channel_features.append([

#             bandpower(
#                 channel,
#                 sf,
#                 (1, 4)
#             ),

#             bandpower(
#                 channel,
#                 sf,
#                 (4, 8)
#             ),

#             bandpower(
#                 channel,
#                 sf,
#                 (8, 12)
#             ),

#             bandpower(
#                 channel,
#                 sf,
#                 (12, 30)
#             )

#         ])

#     features = np.mean(
#         channel_features,
#         axis=0
#     )

#     return features


def extract_features(raw):

    data = raw.get_data()
    sf = raw.info["sfreq"]

    channel_features = []

    # -------------------------
    # STEP 1: BAND POWER PER CHANNEL
    # -------------------------
    for channel in data:

        channel_features.append([
            bandpower(channel, sf, (1, 4)),   # delta
            bandpower(channel, sf, (4, 8)),   # theta
            bandpower(channel, sf, (8, 12)),  # alpha
            bandpower(channel, sf, (12, 30))  # beta
        ])

    channel_features = np.array(channel_features)

    # -------------------------
    # STEP 2: AGGREGATE ACROSS CHANNELS
    # -------------------------
    delta = channel_features[:, 0]
    theta = channel_features[:, 1]
    alpha = channel_features[:, 2]
    beta  = channel_features[:, 3]

    eps = 1e-8

    # -------------------------
    # STEP 3: BUILD 12 FEATURES
    # -------------------------
    features = [

        # Base bands (4)
        np.mean(delta),
        np.mean(theta),
        np.mean(alpha),
        np.mean(beta),

        # Band ratios (3)
        np.mean(delta) / (np.mean(theta) + eps),
        np.mean(alpha) / (np.mean(theta) + eps),
        np.mean(beta) / (np.mean(alpha) + eps),

        # Statistical features (5) across ALL bands
        np.mean(channel_features),
        np.std(channel_features),
        np.var(channel_features),
        np.max(channel_features),
        np.min(channel_features),
    ]

    return np.array(features)
# -------------------------
# BUILD EEG DATASET
# -------------------------
def build_dataset():

    rows = []

    for root, dirs, files in os.walk(
        RAW_DATA_DIR
    ):

        for file in files:

            if not file.endswith(".bdf"):
                continue

            file_path = os.path.join(
                root,
                file
            )

            print(
                f"Processing {file}"
            )

            try:

                raw = preprocess(
                    file_path
                )

                features = extract_features(
                    raw
                )

                label = (
                    1
                    if "sub-pd"
                    in file.lower()
                    else 0
                )

                # rows.append({

                #     "file": file,

                #     "delta":
                #     features[0],

                #     "theta":
                #     features[1],

                #     "alpha":
                #     features[2],

                #     "beta":
                #     features[3],

                #     "label":
                #     label
                # })
                rows.append({

                "file": file,

                "delta": features[0],
                "theta": features[1],
                "alpha": features[2],
                "beta": features[3],

                "delta_theta_ratio": features[4],
                "alpha_theta_ratio": features[5],
                "beta_alpha_ratio": features[6],

                "mean": features[7],
                "std": features[8],
                "variance": features[9],
                "max": features[10],
                "min": features[11],

                "label": label
})
            except Exception as e:

                print(
                    f"Failed: {file}"
                )

                print(e)

    df = pd.DataFrame(rows)

    df.to_csv(
        EEG_FEATURES_PATH,
        index=False
    )

    print("\nSaved:")
    print(
        EEG_FEATURES_PATH
    )

    print(
        f"Samples: {len(df)}"
    )


if __name__ == "__main__":
    build_dataset()