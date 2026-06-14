# import pandas as pd
# from pathlib import Path


# # def load_metadata(path):
# #     """
# #     Load and preprocess participant metadata.
# #     """

# #     df = pd.read_csv(path, sep="\t")

# #     # Create Parkinson label
# #     df["label"] = df["participant_id"].apply(
# #         lambda x: 1 if "pd" in str(x).lower() else 0
# #     )

# #     # Encode gender
# #     if "gender" in df.columns:
# #         df["gender"] = df["gender"].map({
# #             "m": 0,
# #             "f": 1
# #         })

# #     # Fill missing values
# #     df.fillna(-1, inplace=True)

# #     return df


# def load_metadata(path):
#     df = pd.read_csv(path, sep="\t")

#     # Binary Parkinson label
#     df["label"] = df["participant_id"].apply(
#         lambda x: 1 if "pd" in str(x).lower() else 0
#     )

#     # Encode gender
#     if "gender" in df.columns:
#         df["gender"] = (
#             df["gender"]
#             .astype(str)
#             .str.lower()
#             .map({"m": 0, "f": 1})
#         )

#     # Fill missing values by datatype
#     for col in df.columns:

#         if pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(-1)

#         else:
#             df[col] = df[col].fillna("unknown")

#     return df
# def add_season_proxy(df):
#     """
#     Artificial season feature for experimentation.
#     """
#     df["season"] = df.index % 4
#     return df


# if __name__ == "__main__":

#     # Project root directory
#     BASE_DIR = Path(__file__).resolve().parent.parent

#     # Paths
#     raw_file = BASE_DIR / "data" / "raw" / "participants.tsv"
#     processed_file = BASE_DIR / "data" / "processed" / "metadata.csv"

#     print(f"Reading metadata from:\n{raw_file}")

#     if not raw_file.exists():
#         raise FileNotFoundError(
#             f"Metadata file not found:\n{raw_file}"
#         )

#     # Create processed folder if missing
#     processed_file.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     # Process metadata
#     df = load_metadata(raw_file)
#     df = add_season_proxy(df)

#     # Save
#     df.to_csv(processed_file, index=False)

#     print("\nMetadata processed successfully [OK]")
#     print(f"Saved to:\n{processed_file}")
#     print(f"Total participants: {len(df)}")



import pandas as pd
from config import RAW_DATA_DIR, METADATA_PATH

def build_metadata():

    file_path = f"{RAW_DATA_DIR}/participants.tsv"

    df = pd.read_csv(file_path, sep="\t")

    df["label"] = df["participant_id"].apply(
        lambda x: 1 if "pd" in x.lower() else 0
    )

    df["gender"] = df["gender"].map({
        "m": 0,
        "f": 1
    })

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        df[col] = df[col].fillna("unknown")

    df.to_csv(METADATA_PATH, index=False)

    print("Metadata saved:")
    print(METADATA_PATH)

if __name__ == "__main__":
    build_metadata()