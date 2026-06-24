# import os
# import joblib
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier


# OUTPUT_DIR = "outputs/models"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def train_models(X, y):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     models = {
#         "RandomForest": RandomForestClassifier(
#             n_estimators=200,
#             random_state=42
#         ),

#         "SVM": SVC(
#             probability=True
#         ),

#         "LogisticRegression": LogisticRegression(
#             max_iter=1000
#         ),

#         "KNN": KNeighborsClassifier()
#     }

#     results = []

#     for name, model in models.items():

#         print(f"\nTraining {name}")

#         model.fit(X_train, y_train)

#         train_acc = model.score(X_train, y_train)
#         test_acc = model.score(X_test, y_test)

#         results.append({
#             "Model": name,
#             "Train Accuracy": round(train_acc, 4),
#             "Test Accuracy": round(test_acc, 4),
#             "Gap": round(train_acc - test_acc, 4)
#         })

#         model_path = os.path.join(
#             OUTPUT_DIR,
#             f"{name}.pkl"
#         )

#         joblib.dump(model, model_path)

#         print(f"Saved: {model_path}")

#     results_df = pd.DataFrame(results)

#     results_df.to_csv(
#         os.path.join(
#             OUTPUT_DIR,
#             "model_comparison.csv"
#         ),
#         index=False
#     )

#     print("\nModel Comparison")
#     print(results_df)

#     return results_df



# import os
# import joblib
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

# from config import (
#     EEG_FEATURES_PATH,
#     MODELS_DIR
# )

# os.makedirs(
#     MODELS_DIR,
#     exist_ok=True
# )


# def train_models(X, y):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     models = {

#         "RandomForest":
#         RandomForestClassifier(
#             n_estimators=300,
#             max_depth=5,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42
#         ),

#         "SVM":
#         SVC(
#             probability=True
#         ),

#         "LogisticRegression":
#         LogisticRegression(
#             max_iter=1000
#         ),

#         "KNN":
#         KNeighborsClassifier()
#     }

#     results = []

#     trained = {}

#     for name, model in models.items():

#         model.fit(
#             X_train,
#             y_train
#         )

#         train_acc = model.score(
#             X_train,
#             y_train
#         )

#         test_acc = model.score(
#             X_test,
#             y_test
#         )

#         joblib.dump(
#             model,
#             os.path.join(
#                 MODELS_DIR,
#                 f"{name}.pkl"
#             )
#         )

#         trained[name] = model

#         results.append({

#             "Model": name,

#             "Train Accuracy":
#             round(train_acc, 4),

#             "Test Accuracy":
#             round(test_acc, 4),

#             "Gap":
#             round(
#                 train_acc -
#                 test_acc,
#                 4
#             )
#         })

#     pd.DataFrame(
#         results
#     ).to_csv(
#         os.path.join(
#             MODELS_DIR,
#             "model_comparison.csv"
#         ),
#         index=False
#     )

#     return trained


# if __name__ == "__main__":

#     df = pd.read_csv(
#         EEG_FEATURES_PATH
#     )

#     # X = df[
#     #     [
#     #         "delta",
#     #         "theta",
#     #         "alpha",
#     #         "beta"
#     #     ]
#     # ].values
#     FEATURES = [

#         "age",
#         "gender",
#         "hand",
#         "MMSE",
#         "NAART",
#         "disease_duration",
#         "season",

#         "delta",
#         "theta",
#         "alpha",
#         "beta",

#         "delta_theta_ratio",
#         "alpha_theta_ratio",
#         "beta_alpha_ratio",

#         "mean",
#         "std",
#         "variance",
#         "max",
#         "min"
#     ]

#     X = df[
#         FEATURES
#     ].fillna(0)

#     y = df[
#         "label"
#     ].values

#     train_models(X, y)

#     print(
#         "Training Complete"
#     )

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier
)

from sklearn.ensemble import (
    GradientBoostingClassifier
)

from config import (
    FINAL_DATASET_PATH,
    MODELS_DIR
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

FEATURES = [

    "age",
    "gender",
    "hand",
    "MMSE",
    "NAART",
    "disease_duration",
    "rl_deficits",
    "season",

    "delta",
    "theta",
    "alpha",
    "beta",

    "delta_theta_ratio",
    "alpha_theta_ratio",
    "beta_alpha_ratio",

    "mean",
    "std",
    "variance",
    "max",
    "min"
]


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "GradientBoosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "SVM":
        SVC(
            kernel="linear",
            C=0.5,
            probability=True
        ),

        "LogisticRegression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "RidgeClassifier":
        RidgeClassifier()
    }

    results = []

    trained = {}

    for name, model in models.items():

        print(f"\nTraining {name}")

        model.fit(
            X_train,
            y_train
        )

        train_acc = model.score(
            X_train,
            y_train
        )

        test_acc = model.score(
            X_test,
            y_test
        )

        joblib.dump(
            model,
            os.path.join(
                MODELS_DIR,
                f"{name}.pkl"
            )
        )

        trained[name] = model

        results.append({

            "Model": name,

            "Train Accuracy":
            round(train_acc, 4),

            "Test Accuracy":
            round(test_acc, 4),

            "Gap":
            round(
                train_acc - test_acc,
                4
            )
        })

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Test Accuracy : {test_acc:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        os.path.join(
            MODELS_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    print("\nModel Comparison")
    print(results_df)

    return trained


if __name__ == "__main__":

    df = pd.read_csv(
        FINAL_DATASET_PATH
    )
    if "label" not in df.columns:

        if "label_x" in df.columns:
            df["label"] = df["label_x"]

        elif "label_y" in df.columns:
            df["label"] = df["label_y"]

        else:
            raise ValueError(
                f"No label column found.\nColumns={df.columns.tolist()}"
            )

    X = (
        df[FEATURES]
        .fillna(0)
    )

    y = df["label"].values

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        os.path.join(
            MODELS_DIR,
            "scaler.pkl"
        )
    )

    train_models(
        X,
        y
    )

    print(
        "\nTraining Complete"
    )