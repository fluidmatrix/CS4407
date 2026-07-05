from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import pandas as pd
from sklearn.datasets import load_iris


def print_results(test, predicted):
    iris = load_iris()
    for actual, predicted in zip(test[:10], predicted[:10]):
        print(
            "Actual: ", iris.target_names[actual],
            "| Predicted: ", iris.target_names[predicted]
        )
        
def print_heatmap(y_test, y_predict):
    
    # Loading the dataset
    iris = load_iris()
    class_names = iris.target_names

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predict)
    c_matrix = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Classification Report
    report = classification_report(y_test, y_predict, target_names=class_names)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [2, 1]}
    )

    # Heatmap
    sns.heatmap(
        c_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        ax=ax1
    )

    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("Actual Class")
    ax1.set_title("Confusion Matrix")

    # Classification Report
    ax2.axis('off')
    ax2.text(
        0,
        1,
        report,
        fontsize=10,
        family='monospace',
        verticalalignment='top'
    )
    ax2.set_title("Classification Report")

    plt.tight_layout()
    plt.show()