import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
import matplotlib.pyplot as plt

# Data Generation
def generate_synthetic_multiomics_data(n_samples=500, n_features_per_omics=50, random_seed=42):
    np.random.seed(random_seed)
    total_features = n_features_per_omics * 3
    labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    data = np.zeros((n_samples, total_features))
    for i in range(n_samples):
        if labels[i] == 0:
            data[i, :] = np.random.normal(loc=1.0, scale=0.5, size=total_features)
        else:
            data[i, :] = np.random.normal(loc=2.0, scale=0.8, size=total_features)
    feature_names = (
        [f"genomics_{i+1}" for i in range(n_features_per_omics)] +
        [f"proteomics_{i+1}" for i in range(n_features_per_omics)] +
        [f"metabolomics_{i+1}" for i in range(n_features_per_omics)]
    )
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tcga_breast_cancer.csv', index=False)
    print(f"Synthetic data generated and saved to 'data/tcga_breast_cancer.csv' with {n_samples} samples and {total_features} features.")
    return df

# Preprocessing
def preprocess_data(data_path, n_components=10, test_size=0.2, random_seed=42, subsample_size=100):
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    print(f"Explained variance ratio (cumulative): {sum(pca.explained_variance_ratio_):.3f}")
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    if len(X_train) > subsample_size:
        indices = np.random.choice(len(X_train), subsample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    print(f"Subsampled training data to {len(X_train)} samples.")
    return X_train, X_test, y_train, y_test, pca

# Quantum Encoding
def encode_data(features, n_qubits):
    required_length = 2 ** n_qubits
    if len(features) < required_length:
        features = np.pad(features, (0, required_length - len(features)), 'constant')
    else:
        features = features[:required_length]
    norm = np.linalg.norm(features)
    if norm == 0:
        norm = 1e-10
    features = features / norm
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    qc.initialize(features, qr)
    return qc

# QNN Architecture
class QNN:
    def __init__(self, n_qubits, n_layers=1):  # Reduced layers
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = RealAmplitudes(n_qubits, reps=n_layers)
        self.parameters = self.circuit.parameters
        self.simulator = AerSimulator(method='statevector')
        self.basis_gates = ['u1', 'u2', 'u3', 'cx']
        # Transpile the circuit once during initialization
        self.transpiled_circuit = transpile(self.circuit, basis_gates=self.basis_gates, optimization_level=1)

    def forward(self, features, params):
        encoding_circuit = encode_data(features, self.n_qubits)
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(encoding_circuit, inplace=True)
        bound_circuit = self.transpiled_circuit.assign_parameters(params)
        qc.compose(bound_circuit, inplace=True)
        qc.measure_all()
        qc = transpile(qc, self.simulator, basis_gates=self.basis_gates, optimization_level=1)
        job = self.simulator.run(qc, shots=128)  # Further reduced shots
        result = job.result()
        counts = result.get_counts()
        expectation = 0
        for state, count in counts.items():
            if state[0] == '0':
                expectation += count
            else:
                expectation -= count
        expectation /= 128
        return (expectation + 1) / 2

    def compute_loss(self, X, y, params):
        predictions = [self.forward(x, params) for x in X]
        predictions = np.array(predictions)
        epsilon = 1e-10
        loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return loss

# Training
def train_qnn(qnn, X_train, y_train, n_iterations=5):  # Further reduced iterations
    optimizer = SPSA(maxiter=n_iterations)
    params = np.random.random(len(qnn.parameters))
    def objective_function(params):
        return qnn.compute_loss(X_train, y_train, params)
    result = optimizer.minimize(objective_function, params)
    optimized_params = result.x
    return optimized_params

# Evaluation
def evaluate_model(qnn, X_test, y_test, params):
    predictions = [qnn.forward(x, params) for x in X_test]
    predictions_prob = np.array(predictions)
    predictions = predictions_prob > 0.5
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions_prob)
    return accuracy, f1, auc, predictions_prob

# Classical Baseline
def train_classical_baseline(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    predictions_binary = predictions > 0.5
    accuracy = accuracy_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    auc = roc_auc_score(y_test, predictions)
    return accuracy, f1, auc, model.feature_importances_, predictions

# Visualization
def plot_results(qnn_metrics, rf_metrics, feature_importances, y_test, qnn_predictions, rf_predictions):
    results = pd.DataFrame({
        'Model': ['QNN', 'Random Forest'],
        'Accuracy': [qnn_metrics[0], rf_metrics[0]],
        'F1-Score': [qnn_metrics[1], rf_metrics[1]],
        'AUC': [qnn_metrics[2], rf_metrics[2]]
    })
    print("\nResults Table:")
    print(results.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(feature_importances)), feature_importances, tick_label=[f"Feature {i+1}" for i in range(len(feature_importances))])
    plt.title('Feature Importance (Potential Biomarkers)')
    plt.xlabel('PCA Components')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    fpr_qnn, tpr_qnn, _ = roc_curve(y_test, qnn_predictions)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_predictions)
    plt.plot(fpr_qnn, tpr_qnn, label=f'QNN (AUC = {qnn_metrics[2]:.3f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_metrics[2]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main Execution
def main():
    generate_synthetic_multiomics_data()
    X_train, X_test, y_train, y_test, pca = preprocess_data('data/tcga_breast_cancer.csv', n_components=10, subsample_size=100)
    n_qubits = 3  # Reduced qubits
    qnn = QNN(n_qubits=n_qubits, n_layers=1)  # Reduced layers
    optimized_params = train_qnn(qnn, X_train, y_train, n_iterations=5)  # Reduced iterations
    qnn_accuracy, qnn_f1, qnn_auc, qnn_predictions = evaluate_model(qnn, X_test, y_test, optimized_params)
    qnn_metrics = (qnn_accuracy, qnn_f1, qnn_auc)
    print(f"QNN - Accuracy: {qnn_accuracy:.3f}, F1-Score: {qnn_f1:.3f}, AUC: {qnn_auc:.3f}")
    rf_accuracy, rf_f1, rf_auc, feature_importances, rf_predictions = train_classical_baseline(X_train, y_train, X_test, y_test)
    rf_metrics = (rf_accuracy, rf_f1, rf_auc)
    print(f"Random Forest - Accuracy: {rf_accuracy:.3f}, F1-Score: {rf_f1:.3f}, AUC: {rf_auc:.3f}")
    plot_results(qnn_metrics, rf_metrics, feature_importances, y_test, qnn_predictions, rf_predictions)
    print("Plots saved to 'results/feature_importance.png' and 'results/roc_curve.png'.")

if __name__ == "__main__":
    main()