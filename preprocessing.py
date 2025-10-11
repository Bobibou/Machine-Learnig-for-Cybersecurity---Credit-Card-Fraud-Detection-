import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path: str):
    """Charge le dataset depuis un fichier CSV."""
    df = pd.read_csv(path)
    return df

def split_data(df, target='Class', test_size=0.2, random_state=42):
    """Sépare le dataset en jeu d'entraînement et de test."""
    X = df.drop(columns=[target, 'Time'])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def scale_data(X_train, X_test):
    """Normalise les données numériques."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def resample_smote(X, y, ratio=0.2, random_state=42):
    """Applique SMOTE pour équilibrer le dataset."""
    sm = SMOTE(sampling_strategy=ratio, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res