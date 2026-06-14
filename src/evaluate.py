import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

OUTPUT_DIR = "outputs/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def audit_model(
    train_acc,
    test_acc,
    std_acc,
    precision,
    recall,
    f1,
    auc=None
):
    """
    Automatic model health checker.
    """

    warnings = []

    gap = train_acc - test_acc

    # ------------------------
    # Overfitting Detection
    # ------------------------
    if train_acc > 0.95 and gap > 0.10:
        warnings.append(
            "SEVERE OVERFITTING: model memorizes training data."
        )

    elif gap > 0.05:
        warnings.append(
            "MODERATE OVERFITTING: train-test gap is high."
        )

    # ------------------------
    # Underfitting Detection
    # ------------------------
    if train_acc < 0.70 and test_acc < 0.70:
        warnings.append(
            "UNDERFITTING: model cannot learn patterns."
        )

    # ------------------------
    # High Variance
    # ------------------------
    if std_acc > 0.08:
        warnings.append(
            "UNSTABLE MODEL: high Monte Carlo variance."
        )

    # ------------------------
    # Low Precision
    # ------------------------
    if precision < 0.70:
        warnings.append(
            "LOW PRECISION: too many false positives."
        )

    # ------------------------
    # Low Recall
    # ------------------------
    if recall < 0.70:
        warnings.append(
            "LOW RECALL: too many false negatives."
        )

    # ------------------------
    # Low F1
    # ------------------------
    if f1 < 0.70:
        warnings.append(
            "LOW F1 SCORE: poor class balance."
        )

    # ------------------------
    # Poor AUC
    # ------------------------
    if auc is not None:

        if auc < 0.60:
            warnings.append(
                "VERY POOR SEPARATION: AUC below 0.60."
            )

        elif auc < 0.75:
            warnings.append(
                "WEAK CLASS SEPARATION."
            )

    # ------------------------
    # Final Verdict
    # ------------------------
    if len(warnings) == 0:

        verdict = (
            "HEALTHY MODEL: no major issues detected."
        )

    elif len(warnings) <= 2:

        verdict = (
            "ACCEPTABLE MODEL: minor warnings."
        )

    else:

        verdict = (
            "POOR MODEL: multiple issues detected."
        )

    return verdict, warnings

def monte_carlo(models, X, y, runs=20):

    results = []

    print("\n" + "=" * 60)
    print("MONTE CARLO VALIDATION")
    print("=" * 60)

    for name, model in models.items():

        train_scores = []
        test_scores = []

        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []

        print(f"\nEvaluating {name}")

        for run in range(runs):

            idx = np.random.permutation(len(X))

            split = int(0.8 * len(X))

            train_idx = idx[:split]
            test_idx = idx[split:]

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            current_model = clone(model)

            current_model.fit(X_train, y_train)

            train_pred = current_model.predict(X_train)
            test_pred = current_model.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            train_scores.append(train_acc)
            test_scores.append(test_acc)

            precision_scores.append(
                precision_score(
                    y_test,
                    test_pred,
                    zero_division=0
                )
            )

            recall_scores.append(
                recall_score(
                    y_test,
                    test_pred,
                    zero_division=0
                )
            )

            f1_scores.append(
                f1_score(
                    y_test,
                    test_pred,
                    zero_division=0
                )
            )

            try:
                probs = current_model.predict_proba(X_test)[:, 1]

                auc_scores.append(
                    roc_auc_score(
                        y_test,
                        probs
                    )
                )

            except Exception:
                pass

        train_mean = np.mean(train_scores)
        test_mean = np.mean(test_scores)

        gap = train_mean - test_mean

        # ----------------------
        # Fit Diagnosis
        # ----------------------
        if train_mean > 0.95 and gap > 0.10:
            status = "SEVERE OVERFIT"

        elif gap > 0.05:
            status = "MODERATE OVERFIT"

        elif train_mean < 0.70 and test_mean < 0.70:
            status = "UNDERFIT"

        else:
            status = "GOOD GENERALIZATION"

        model_result = {
            "Model": name,

            "Train Accuracy Mean": round(train_mean, 4),
            "Train Accuracy Std": round(np.std(train_scores), 4),

            "Test Accuracy Mean": round(test_mean, 4),
            "Test Accuracy Std": round(np.std(test_scores), 4),

            "Generalization Gap": round(gap, 4),

            "Precision": round(np.mean(precision_scores), 4),
            "Recall": round(np.mean(recall_scores), 4),
            "F1 Score": round(np.mean(f1_scores), 4),

            "ROC AUC": round(np.mean(auc_scores), 4)
            if len(auc_scores) > 0 else None,

            "Status": status
        }

        results.append(model_result)

        print(f"Status: {status}")
        print(f"Train Accuracy: {train_mean:.4f}")
        print(f"Test Accuracy : {test_mean:.4f}")
        print(f"Gap           : {gap:.4f}")

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(
        OUTPUT_DIR,
        "monte_carlo_results.csv"
    )

    results_df.to_csv(csv_path, index=False)

    json_path = os.path.join(
        OUTPUT_DIR,
        "monte_carlo_results.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print("\nSaved:")
    print(csv_path)
    print(json_path)

    return results_df