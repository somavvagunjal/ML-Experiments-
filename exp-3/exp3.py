from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD THE ENGINEERING DATASET
# ============================================================

data = pd.read_csv(Path(__file__).with_name("industrial_compressor_data.csv"))

print("ENGINEERING DATASET")
print(data.head())

# ============================================================
# 2. IDENTIFY SAMPLES, FEATURES AND TARGET
# ============================================================

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

# Number of samples
print("\nNumber of Samples:", data.shape[0])

# Define features and target based on the actual columns in industrial_compressor_data.csv
features = [
    "Inlet_Pressure_bar",
    "Outlet_Pressure_bar",
    "Air_Flow_Rate_m3_min",
    "Inlet_Temperature_C",
    "Motor_Speed_RPM",
    "Power_Consumption_kW"
]

target = "Compressor_Efficiency_pct" # Updated target variable

X = data[features]
y = data[target]

print("\nFeatures:")
print(features)

print("\nTarget Variable:")
print(target)

print("\nTarget Values:")
print(y.unique())

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDESCRIPTIVE STATISTICS")
print(data[features].describe())

# ============================================================
# 4. DETECT MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nTotal Missing Values:")
print(data.isnull().sum().sum())

# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# For regression, stratify should not be used as it's for classification problems.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDATA SPLITTING")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easy viewing
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features,
    index=X_test.index
)

print("\nSCALED TRAINING FEATURES")
print(X_train_scaled.head())

print("\nSCALED TESTING FEATURES")
print(X_test_scaled.head())

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Total Samples:", data.shape[0])
print("Total Features:", len(features))
print("Features:", features)
print("Target:", target)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

print("Total Missing Values:", data.isnull().sum().sum())