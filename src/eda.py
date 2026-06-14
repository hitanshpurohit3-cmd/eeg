import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/processed/metadata.csv"
OUTPUT_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns)


# ----------------------------
# BASIC INFO
# ----------------------------
print("\nMissing Values:\n", df.isnull().sum())
print("\nLabel Distribution:\n", df["label"].value_counts())


# ----------------------------
# 1. LABEL DISTRIBUTION
# ----------------------------
plt.figure()
sns.countplot(x="label", data=df)
plt.title("Parkinson vs Healthy Control Distribution")
plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
plt.close()


# ----------------------------
# 2. GENDER DISTRIBUTION
# ----------------------------
plt.figure()
sns.countplot(x="gender", hue="label", data=df)
plt.title("Gender vs Disease Label")
plt.savefig(f"{OUTPUT_DIR}/gender_vs_label.png")
plt.close()


# ----------------------------
# 3. AGE DISTRIBUTION
# ----------------------------
plt.figure()
sns.histplot(data=df, x="age", hue="label", kde=True, bins=20)
plt.title("Age Distribution by Disease Status")
plt.savefig(f"{OUTPUT_DIR}/age_distribution.png")
plt.close()


# ----------------------------
# 4. SEASON PROXY ANALYSIS
# ----------------------------
# plt.figure()
# sns.countplot(x="season", hue="label", data=df)
# plt.title("Temporal Cohort (Season Proxy) vs Label")
# plt.savefig(f"{OUTPUT_DIR}/season_vs_label.png")
# plt.close()


# ----------------------------
# 5. AGE vs MMSE (if exists)
# ----------------------------
if "mmse" in df.columns:
    plt.figure()
    sns.scatterplot(x="age", y="mmse", hue="label", data=df)
    plt.title("Age vs Cognitive Score (MMSE)")
    plt.savefig(f"{OUTPUT_DIR}/age_vs_mmse.png")
    plt.close()


# ----------------------------
# 6. CORRELATION HEATMAP
# ----------------------------
plt.figure(figsize=(10, 6))

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()


# ----------------------------
# 7. FEATURE-WISE GROUP COMPARISON
# ----------------------------
features = ["age", "gender", "season"]

for col in features:
    if col in df.columns:
        plt.figure()
        sns.boxplot(x="label", y=col, data=df)
        plt.title(f"{col} Distribution by Label")
        plt.savefig(f"{OUTPUT_DIR}/{col}_boxplot.png")
        plt.close()


# ----------------------------
# 8. MULTIVARIATE PAIRPLOT (sample only)
# ----------------------------
sample_cols = [c for c in ["age", "gender", "season", "label"] if c in df.columns]

if len(sample_cols) >= 3:
    sns.pairplot(df[sample_cols], hue="label")
    plt.savefig(f"{OUTPUT_DIR}/pairplot.png")
    plt.close()


print("\nEDA COMPLETE — all plots saved to outputs/plots [OK]")