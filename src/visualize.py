import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(corr):
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

def plot_model_scores(scores):
    plt.bar(scores.keys(), scores.values())
    plt.title("Model Comparison")
    plt.show()