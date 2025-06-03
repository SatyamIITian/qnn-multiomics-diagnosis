# src/run_analysis.py

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score, roc_curve
import shap
from data_generator import load_data, get_feature_names
from preprocessor import Preprocessor
from hybrid_model import QGAHybridModel, tune_hybrid_model
from classical_baseline import RandomForestBaseline, tune_rf_model
from visualizer import Visualizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Run analysis for QNN-Multiomics-Diagnosis")
    parser.add_argument("--data_type", type=str, default="synthetic", choices=["synthetic", "real"],
                        help="Type of data to use: 'synthetic' or 'real'")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load data
    print(f"Loading {args.data_type} data...")
    X, y = load_data(data_type=args.data_type)
    feature_names = get_feature_names(data_type=args.data_type)

    print(f"Dataset shape: {X.shape} (samples: {X.shape[0]}, features: {X.shape[1]})")
    if y is not None:
        print(f"Label distribution: {np.bincount(y)} (0: {np.sum(y == 0)}, 1: {np.sum(y == 1)})")
    else:
        print("Labels are missing in the dataset.")

    # Preprocess data
    preprocessor = Preprocessor()
    X_scaled = preprocessor.fit_transform(X, y=y)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(X))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_train = y[train_idx] if y is not None else None
    y_test = y[test_idx] if y is not None else None

    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    omics_data_train = preprocessor.split_omics(X_train, data_type=args.data_type)
    omics_data_test = preprocessor.split_omics(X_test, data_type=args.data_type)

    visualizer = Visualizer()

    if y_train is not None:
        print("Tuning QGA-Hybrid model hyperparameters...")
        best_params_hybrid = tune_hybrid_model(omics_data_train, y_train, omics_data_test, y_test)
        print(f"Best hyperparameters for QGA-Hybrid model: {best_params_hybrid}")

        print("Training QGA-Hybrid model with best hyperparameters...")
        hybrid_model = QGAHybridModel(
            input_dims=[50, 50, 50],
            hidden_dim=best_params_hybrid["hidden_dim"],
            num_heads=best_params_hybrid["num_heads"],
            dropout_rate=best_params_hybrid["dropout_rate"]
        )
        hybrid_model.fit(
            omics_data_train, y_train,
            epochs=best_params_hybrid["epochs"],
            batch_size=best_params_hybrid["batch_size"],
            lr=best_params_hybrid["lr"],
            alpha=best_params_hybrid["alpha"]
        )
        y_pred_hybrid = hybrid_model.predict(omics_data_test)
        y_pred_proba_hybrid = hybrid_model.predict_proba(omics_data_test)

        auc_hybrid = roc_auc_score(y_test, y_pred_proba_hybrid)
        accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)
        precision_hybrid = precision_score(y_test, y_pred_hybrid)
        recall_hybrid = recall_score(y_test, y_pred_hybrid)
        f1_hybrid = f1_score(y_test, y_pred_hybrid)
        mcc_hybrid = matthews_corrcoef(y_test, y_pred_hybrid)
        balanced_acc_hybrid = balanced_accuracy_score(y_test, y_pred_hybrid)
        cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
        tn_hybrid, fp_hybrid, fn_hybrid, tp_hybrid = cm_hybrid.ravel()

        print("\nQGA-Hybrid Model Performance (real data):")
        print(f"AUC: {auc_hybrid:.3f}")
        print(f"Accuracy: {accuracy_hybrid:.3f}")
        print(f"Balanced Accuracy: {balanced_acc_hybrid:.3f}")
        print(f"Precision: {precision_hybrid:.3f}")
        print(f"Recall: {recall_hybrid:.3f}")
        print(f"F1-Score: {f1_hybrid:.3f}")
        print(f"Matthews Correlation Coefficient: {mcc_hybrid:.3f}")
        print("Confusion Matrix:")
        print(f"True Negatives (TN): {tn_hybrid}")
        print(f"False Positives (FP): {fp_hybrid}")
        print(f"False Negatives (FN): {fn_hybrid}")
        print(f"True Positives (TP): {tp_hybrid}")

        print("\nTuning random forest baseline hyperparameters...")
        best_params_rf = tune_rf_model(X_train, y_train, X_test, y_test)
        print(f"Best hyperparameters for random forest: {best_params_rf}")

        print("Training random forest baseline with best hyperparameters...")
        rf_baseline = RandomForestBaseline()
        rf_baseline.fit(
            X_train, y_train,
            n_estimators=best_params_rf["n_estimators"],
            max_depth=best_params_rf["max_depth"],
            min_samples_split=best_params_rf["min_samples_split"]
        )
        y_pred_rf = rf_baseline.predict(X_test)
        y_pred_proba_rf = rf_baseline.predict_proba(X_test)

        auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        mcc_rf = matthews_corrcoef(y_test, y_pred_rf)
        balanced_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

        print("\nRandom Forest Performance (real data):")
        print(f"AUC: {auc_rf:.3f}")
        print(f"Accuracy: {accuracy_rf:.3f}")
        print(f"Balanced Accuracy: {balanced_acc_rf:.3f}")
        print(f"Precision: {precision_rf:.3f}")
        print(f"Recall: {recall_rf:.3f}")
        print(f"F1-Score: {f1_rf:.3f}")
        print(f"Matthews Correlation Coefficient: {mcc_rf:.3f}")
        print("Confusion Matrix:")
        print(f"True Negatives (TN): {tn_rf}")
        print(f"False Positives (FP): {fp_rf}")
        print(f"False Negatives (FN): {fn_rf}")
        print(f"True Positives (TP): {tp_rf}")

        feature_importances = rf_baseline.get_feature_importance()
        top_indices = np.argsort(feature_importances)[::-1][:5]
        print("\nTop 5 Most Important Features (Random Forest):")
        for idx in top_indices:
            print(f"Feature {feature_names[idx]}: Importance = {feature_importances[idx]:.4f}")

        print("\nComputing SHAP values for random forest...")
        explainer = shap.TreeExplainer(rf_baseline.model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
        plt.savefig(f"results/shap_summary_{args.data_type}.png", dpi=300, bbox_inches='tight')
        plt.close()

        visualizer.plot_roc_curve(y_test, y_pred_proba_hybrid, y_pred_proba_rf, f"results/roc_curve_{args.data_type}.png")
        visualizer.plot_confusion_matrix(y_test, y_pred_hybrid, f"results/confusion_matrix_hybrid_{args.data_type}.png")
        visualizer.plot_confusion_matrix(y_test, y_pred_rf, f"results/confusion_matrix_rf_{args.data_type}.png")
        visualizer.plot_feature_importance(feature_importances, feature_names, f"results/feature_importance_rf_{args.data_type}.png")
        visualizer.plot_feature_distribution(X_scaled, feature_names, f"results/feature_distribution_{args.data_type}.png")
    else:
        print("Labels are missing. Running in unsupervised mode...")
        hybrid_model = QGAHybridModel()
        hybrid_model.fit(omics_data_train, y_train)
        visualizer.plot_feature_distribution(X_scaled, feature_names, f"results/feature_distribution_{args.data_type}.png")
        print(f"Feature distribution plot saved to results/feature_distribution_{args.data_type}.png")

if __name__ == "__main__":
    main()