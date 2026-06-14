# # =========================
# # 🧠 Parkinson EEG Pipeline - Main Runner
# # =========================

# import os
# import subprocess
# from src import config

# # -------------------------
# # Utility Runner
# # -------------------------
# def run_script(script_path, description):
#     print("\n" + "="*60)
#     print(f"🚀 Running: {description}")
#     print("="*60)

#     result = subprocess.run(["python", script_path], capture_output=True, text=True)

#     if result.returncode == 0:
#         print(result.stdout)
#         print(f"✔ Completed: {description}")
#     else:
#         print("❌ Error in:", description)
#         print(result.stderr)


# # -------------------------
# # Ensure directories exist
# # -------------------------
# def create_dirs():
#     os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
#     os.makedirs(config.PLOTS_DIR, exist_ok=True)
#     os.makedirs(config.MODELS_DIR, exist_ok=True)


# # -------------------------
# # Pipeline Steps
# # -------------------------
# PIPELINE = [
#     ("src/dataset_builder.py", "Dataset Builder (metadata processing)"),
#     ("src/eda.py", "Exploratory Data Analysis"),
#     ("src/preprocessing.py", "EEG Preprocessing"),
#     ("src/feature_extraction.py", "Feature Extraction"),
#     ("src/train.py", "Model Training"),
#     ("src/evaluate.py", "Monte Carlo Evaluation"),
# ]


# # -------------------------
# # MAIN
# # -------------------------
# if __name__ == "__main__":

#     print("\n🧠 ===============================")
#     print("   Parkinson EEG Research Pipeline")
#     print("===============================\n")

#     create_dirs()

#     for script, desc in PIPELINE:
#         run_script(script, desc)

#     print("\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
#     print("📊 Check outputs/plots and outputs/models for results")



# =========================
# Parkinson EEG Pipeline - Main Runner
# =========================

import os
import sys
import subprocess
from src import config


# -------------------------
# Utility Runner
# -------------------------
def run_script(script_path, description):

    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print("=" * 60)

    print(f"Python: {sys.executable}")
    print(f"Script: {script_path}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:

        if result.stdout:
            print(result.stdout)

        print(f"[SUCCESS] {description}")
        return True

    else:

        print(f"[FAILED] {description}")

        if result.stderr:
            print("\nError Details:")
            print(result.stderr)

        return False


# -------------------------
# Ensure directories exist
# -------------------------
def create_dirs():

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)


# -------------------------
# Pipeline Steps
# -------------------------
# PIPELINE = [
#     ("src/dataset_builder.py", "Dataset Builder"),
#     ("src/eda.py", "Exploratory Data Analysis"),
#     ("src/preprocessing.py", "EEG Preprocessing"),
#     ("src/feature_extraction.py", "Feature Extraction"),
#     ("src/train.py", "Model Training"),
#     ("src/evaluate.py", "Monte Carlo Evaluation"),
# ]
PIPELINE = [

    (
        "src/dataset_builder.py",
        "Metadata Builder"
    ),

    (
        "src/feature_builder.py",
        "EEG Feature Builder"
    ),

    (
        "src/eda.py",
        "EDA"
    ),

    (
        "src/train.py",
        "Training"
    ),

    (
        "src/evaluate.py",
        "Evaluation"
    )
]

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Parkinson EEG Research Pipeline")
    print("=" * 60)

    print(f"\nUsing Python:")
    print(sys.executable)

    create_dirs()

    for script, description in PIPELINE:

        success = run_script(script, description)

        if not success:
            print("\nPipeline stopped due to error.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nOutputs:")
    print(f"Plots  : {config.PLOTS_DIR}")
    print(f"Models : {config.MODELS_DIR}")