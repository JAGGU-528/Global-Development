import pandas as pd
import numpy as np
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
df = pd.read_excel('World_development_mesurement.xlsx')

# =========================
# REMOVE SPECIAL CHARACTERS
# Only these 5 columns have mixed chars ($ % ,)
# =========================
cols = ['Business Tax Rate', 'GDP', 'Health Exp/Capita',
        'Tourism Inbound', 'Tourism Outbound']

for col in cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# DROP COLUMNS
# =========================
# High missing (>50%): Ease of Business
df.drop('Ease of Business', axis=1, inplace=True)

# Single-value column, irrelevant for modelling
df.drop('Number of Records', axis=1, inplace=True)

# Country is non-numeric — drop before imputing/scaling
# but save the column order of features BEFORE dropping Country
df.drop('Country', axis=1, inplace=True)

# =========================
# SAVE FEATURE NAMES
# (saved BEFORE imputing so it reflects the actual model input columns)
# =========================
features = df.columns.tolist()

# =========================
# STEP 1: IMPUTE (median) — fit on raw numeric data
# =========================
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# =========================
# STEP 2: LOG TRANSFORMATION — applied after imputing
# Reduces right-skew and dampens extreme outliers
# =========================
log_cols = ['Business Tax Rate', 'Hours to do Tax', 'Tourism Outbound',
            'Energy Usage', 'GDP', 'Health Exp/Capita',
            'Tourism Inbound', 'CO2 Emissions',
            'Days to Start Business', 'Population Total',
            'Lending Interest']

for col in log_cols:
    df_imputed[col] = np.log1p(df_imputed[col])

# =========================
# STEP 3: SCALE — fit on log-transformed data
# =========================
scaler = StandardScaler()
scaler.fit(df_imputed)

# =========================
# SAVE ARTIFACTS TO model/ FOLDER
# =========================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb") as f:
    pickle.dump(imputer, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f:
    pickle.dump(features, f)

print("✅ imputer.pkl saved")
print("✅ scaler.pkl saved")
print("✅ features.pkl saved")
print(f"✅ Features ({len(features)}):", features)
