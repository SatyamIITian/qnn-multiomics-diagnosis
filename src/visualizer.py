# src/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class Visualizer:
    def __init__(self):
        sns.set_theme()

    def plot_feature_distribution(self, X, feature_names, save_path):
        plt.figure(figsize=(12, 6))
        for i in range(min(5, X.shape[1])):
            sns.histplot(X[:, i], kde=True, label=feature_names[i], alpha=0.5)
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.title('Feature Distribution (First 5 Features)')
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba_hybrid, y_pred_proba_rf, save_path):
        from sklearn.metrics import roc_curve, auc
        fpr_hybrid, tpr_hybrid, _ = roc_curve(y_true, y_pred_proba_hybrid)
        roc_auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
        
        fpr_rf, tpr_rf, _ = roc_curve(y_true, y_pred_proba_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        plt.figure()
        plt.plot(fpr_hybrid, tpr_hybrid, label=f'QGA-Hybrid Model (AUC = {roc_auc_hybrid:.2f})')
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, importances, feature_names, save_path):
        indices = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importances[indices], align='center')
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()