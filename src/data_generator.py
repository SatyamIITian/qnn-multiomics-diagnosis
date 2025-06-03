# src/data_generator.py

import pandas as pd
import numpy as np
import os

def load_data(data_type="synthetic"):
    """
    Load data based on the specified type.
    
    Args:
        data_type (str): Type of data ('synthetic' or 'real')
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    if data_type == "real":
        file_path = os.path.join("data", "tcga_breast_cancer_real.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found.")
        
        df = pd.read_csv(file_path)
        if "label" not in df.columns:
            print("Labels are missing in the real dataset.")
            X = df.drop(columns=["label"], errors="ignore").values
            y = None
        else:
            X = df.drop(columns=["label"]).values
            y = df["label"].values
        return X, y
    else:
        raise ValueError("Only 'real' data type is supported in this setup.")

def get_feature_names(data_type="synthetic"):
    """
    Get feature names based on the data type.
    
    Args:
        data_type (str): Type of data ('synthetic' or 'real')
    
    Returns:
        list: List of feature names
    """
    if data_type == "real":
        file_path = os.path.join("data", "tcga_breast_cancer_real.csv")
        df = pd.read_csv(file_path)
        feature_names = df.drop(columns=["label"], errors="ignore").columns.tolist()
        return feature_names
    else:
        raise ValueError("Only 'real' data type is supported in this setup.")