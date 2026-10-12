# EXPERIMENT 5  

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# LOAD THE ENGINEERING DATASET
# ============================================================

base_dir = Path(__file__).resolve().parent
candidate_paths = [
    base_dir / "industrial_machine_enery_data.csv",
    base_dir / "industrial_machine_energy_data.csv",
]

data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        "Could not find the machine energy dataset. Looked for: "
        + ", ".join(str(p.name) for p in candidate_paths)
    )

data = pd.read_csv(data_path)

print("ENGINEERING DATASET")
print(data.head())

# ============================================================
# Identify features
# ============================================================

features = [
    "Machine_Load_pct",
    "Motor_Speed_RPM",
    "Operating_Temperature_C",
    "Production_Rate_units_hr",
]
target = "Energy_Consumption_kWh"

print("Features:", features)
print("Target:", target)

# ============================================================
# separate
# ============================================================

X = data[features]
y = data[target]

# ============================================================
# split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ============================================================
# train
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================
# display
# ============================================================

results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nACTUAL VS PREDICTED ENERGY CONSUMPTION")
print(results.head())

print("\nMODEL PERFORMANCE")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} kWh")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.3f}")

print("\nMODEL COEFFICIENTS")
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")
print(f"Intercept: {model.intercept_:.4f}")