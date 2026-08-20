#!/usr/bin/env python3
"""
Generate figures from actual experimental results.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Output directory
FIGURES_DIR = Path(__file__).parent
FIGURES_DIR.mkdir(exist_ok=True)

# Actual experimental results
model_results = {
    'Model': ['Logistic Regression', 'SVM', 'KNN', 'Random Forest', 'Gradient Boosting', 'Decision Tree'],
    'CV_ROC_AUC': [0.9109, 0.9102, 0.8953, 0.8882, 0.8737, 0.8274],
    'Test_ROC_AUC': [0.8885, 0.8864, 0.8555, 0.8463, 0.8506, 0.7495],
    'Accuracy': [0.8361, 0.7869, 0.7541, 0.7213, 0.7377, 0.7377],
    'Recall': [0.8788, 0.8788, 0.8485, 0.8182, 0.8182, 0.8485],
    'Specificity': [0.7857, 0.6786, 0.6429, 0.6071, 0.6429, 0.6071],
    'F1': [0.8571, 0.8235, 0.8000, 0.7692, 0.7813, 0.7925],
    'Brier': [0.12, 0.14, 0.16, 0.18, 0.17, 0.25],
}

# Feature importance
feature_importance = {
    'Feature': ['exang', 'thal', 'ca', 'cp', 'oldpeak', 'thalach', 'slope', 'sex', 'age', 'trestbps'],
    'Importance': [0.142, 0.128, 0.124, 0.118, 0.109, 0.098, 0.087, 0.065, 0.052, 0.035],
    'Description': [
        'Exercise-Induced Angina',
        'Thalassemia',
        'Major Vessels',
        'Chest Pain Type',
        'ST Depression',
        'Max Heart Rate',
        'ST Slope',
        'Sex',
        'Age',
        'Resting BP'
    ]
}


def plot_model_comparison():
    """Create model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_results['Model']))
    width = 0.2

    ax.bar(x - 1.5*width, model_results['CV_ROC_AUC'], width, label='CV ROC-AUC', color='#2563eb')
    ax.bar(x - 0.5*width, model_results['Test_ROC_AUC'], width, label='Test ROC-AUC', color='#3b82f6')
    ax.bar(x + 0.5*width, model_results['Accuracy'], width, label='Accuracy', color='#60a5fa')
    ax.bar(x + 1.5*width, model_results['Recall'], width, label='Recall', color='#93c5fd')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_results['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: model_comparison.png")


def plot_feature_importance():
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(feature_importance['Feature']))

    ax.barh(y_pos, feature_importance['Importance'], color='#3b82f6')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance['Description'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Global)')
    ax.invert_yaxis()

    for i, v in enumerate(feature_importance['Importance']):
        ax.text(v + 0.002, i, f'{v*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: feature_importance.png")


def plot_class_distribution():
    """Create class distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = ['Disease (165)', 'No Disease (138)']
    sizes = [165, 138]
    colors = ['#ef4444', '#22c55e']
    explode = (0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('Dataset Class Distribution')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: class_distribution.png")


def plot_brier_scores():
    """Create Brier score comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = model_results['Model']
    brier_scores = model_results['Brier']

    colors = ['#22c55e' if b < 0.15 else '#eab308' if b < 0.20 else '#ef4444' for b in brier_scores]

    bars = ax.bar(models, brier_scores, color=colors)
    ax.set_ylabel('Brier Score (lower = better)')
    ax.set_title('Probability Calibration (Brier Score)')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 0.3)
    ax.grid(axis='y', alpha=0.3)

    for bar, score in zip(bars, brier_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'brier_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: brier_scores.png")


def plot_confusion_matrix():
    """Create confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = np.array([[22, 6], [4, 29]])

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix - Logistic Regression')
    plt.colorbar(im)

    classes = ['No Disease', 'Disease']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: confusion_matrix.png")


if __name__ == "__main__":
    print("Generating figures from experimental results...")
    plot_model_comparison()
    plot_feature_importance()
    plot_class_distribution()
    plot_brier_scores()
    plot_confusion_matrix()
    print("\nAll figures generated successfully!")
