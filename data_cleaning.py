"""
Week 2 - Data Collection, Cleaning and Preprocessing
Logistics Data Analyst Internship

This script demonstrates a complete preprocessing workflow:
1. Load the raw logistics dataset
2. Inspect the data
3. Remove duplicate rows
4. Handle missing numerical and categorical values
5. Convert date columns to datetime
6. Create useful logistics features
7. Save the cleaned dataset

The dataset used for this internship project is simulated logistics data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_FILE = DATA_DIR / "logistics_raw_simulated.csv"
CLEAN_FILE = DATA_DIR / "logistics_cleaned.csv"


# ---------------------------------------------------------
# 2. Load the raw dataset
# ---------------------------------------------------------

print("Loading raw logistics dataset...")

df = pd.read_csv(RAW_FILE)

print("\nDataset shape before cleaning:", df.shape)
print("\nFirst five records:")
print(df.head())


# ---------------------------------------------------------
# 3. Inspect data types and missing values
# ---------------------------------------------------------

print("\nData types:")
print(df.dtypes)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows before cleaning:", df.duplicated().sum())


# ---------------------------------------------------------
# 4. Remove duplicate records
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates().copy()

print("\nDuplicate rows removed:", duplicate_count)


# ---------------------------------------------------------
# 5. Convert date columns
# ---------------------------------------------------------

date_columns = [
    "Order_Date",
    "Shipping_Date"
]

for column in date_columns:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")


# ---------------------------------------------------------
# 6. Handle missing numerical values
# ---------------------------------------------------------

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numerical_columns:
    if df[column].isnull().any():
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)


# ---------------------------------------------------------
# 7. Handle missing categorical values
# ---------------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    if df[column].isnull().any():
        mode_values = df[column].mode(dropna=True)

        if len(mode_values) > 0:
            df[column] = df[column].fillna(mode_values.iloc[0])
        else:
            df[column] = df[column].fillna("Unknown")


# ---------------------------------------------------------
# 8. Feature engineering
# ---------------------------------------------------------

# Calculate actual shipping days if the required columns exist.
if "Order_Date" in df.columns and "Shipping_Date" in df.columns:
    df["Actual_Shipping_Days"] = (
        df["Shipping_Date"] - df["Order_Date"]
    ).dt.days


# Create delivery delay if scheduled and actual shipping days exist.
if "Scheduled_Days" in df.columns and "Actual_Shipping_Days" in df.columns:
    df["Delay_Days"] = (
        df["Actual_Shipping_Days"] - df["Scheduled_Days"]
    )

    # A shipment is considered late when delay is greater than zero.
    df["Late_Flag"] = (
        df["Delay_Days"] > 0
    ).astype(int)


# ---------------------------------------------------------
# 9. Basic data-quality checks
# ---------------------------------------------------------

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:", df.duplicated().sum())

if "Late_Flag" in df.columns:
    print("\nLate delivery distribution:")
    print(df["Late_Flag"].value_counts())


# ---------------------------------------------------------
# 10. Save cleaned dataset
# ---------------------------------------------------------

df.to_csv(CLEAN_FILE, index=False)

print("\nCleaned dataset saved to:")
print(CLEAN_FILE)

print("\nFinal dataset shape:", df.shape)

print("\nData cleaning completed successfully.")
