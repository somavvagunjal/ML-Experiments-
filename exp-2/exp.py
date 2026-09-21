# -----------------------------------------------
# EXPERIMENT 2 - Fixed for VS Code
# -----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Auto-find csv in same folder as exp.py - no error
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "rotating_machine_data.csv")

data = pd.read_csv(csv_path)

print("Rotating Machine Dataset")
print(data.head())

print("\nDataset size:", data.shape)
print("\nColumn names:")
print(data.columns)
print("\nData information:")
print(data.info())

inputs = [
    "Speed_RPM",
    "Vibration_mm_s",
    "Bearing_Temperature_C",
    "Motor_Current_A",
    "Acoustic_dB"
]
output = "Machine_Health"

print("\nInput Variables:")
print(inputs)
print("\nOutput Variable:")
print(output)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSensor Statistics:")
print(data[inputs].describe())

data[inputs].plot(figsize=(10, 6), subplots=True)
plt.suptitle("Rotating Machine Sensor Readings")
plt.tight_layout()
plt.show()

# Prepare the data
data[inputs] = data[inputs].fillna(data[inputs].mean())

data["Machine_Health"] = data["Machine_Health"].map({
    "Healthy": 0,
    "Faulty": 1
})

scaler = StandardScaler()
data[inputs] = scaler.fit_transform(data[inputs])

print("\nPrepared Data:")
print(data.head())
print("\nData is ready for Machine Learning.")