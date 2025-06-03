# src/preprocessor.py

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = None

    def fit_transform(self, X, y=None):
        """
        Fit the scaler and selector to the data and transform it.
        
        Args:
            X (np.ndarray): Feature matrix (n_samples × n_features)
            y (np.ndarray, optional): Labels for supervised feature selection
        
        Returns:
            np.ndarray: Transformed feature matrix
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection: Select top 150 features (50 per omics type)
        if y is not None:
            self.selector = SelectKBest(score_func=mutual_info_classif, k=150)
            X_selected = self.selector.fit_transform(X_scaled, y)
            return X_selected
        return X_scaled

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
            return X_selected
        return X_scaled

    def split_omics(self, X, data_type="synthetic"):
        """
        Split the data into omics types.
        
        Args:
            X (np.ndarray): Feature matrix
            data_type (str): Type of data ('synthetic' or 'real')
        
        Returns:
            list: List of omics types
        """
        if data_type == "real":
            omics_split = (50, 50, 50)  # After feature selection, 150 features → 50 per type
        else:
            raise ValueError("Only 'real' data type is supported in this setup.")
        
        if sum(omics_split) != X.shape[1]:
            raise ValueError(f"Sum of omics_split {sum(omics_split)} does not match number of features {X.shape[1]}")
        
        start = 0
        omics_data = []
        for size in omics_split:
            end = start + size
            omics_data.append(X[:, start:end])
            start = end
        return omics_data