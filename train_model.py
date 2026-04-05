import pandas as pd
import numpy as np
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATA (RELATIVE PATH)
# ===============================
print("Loading data...")
df = pd.read_excel('World_development_mesurement.xlsx')
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# ===============================
# REMOVE SPECIAL CHARACTERS
# ===============================
print("\nCleaning currency and percentage symbols...")
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

print("✅ Special characters removed")

# ===============================
# DROP HIGH MISSING COLUMN
# ===============================
print("\nDropping 'Ease of Business' (high missing rate)...")
df.drop('Ease of Business', axis=1, inplace=True)

# ===============================
# DROP IRRELEVANT COLUMN
# ===============================
print("Dropping 'Number of Records' (metadata column)...")
df.drop('Number of Records', axis=1, inplace=True)

# ===============================
# DROP COUNTRY (NON-NUMERIC)
# ===============================
print("Dropping 'Country' (non-numeric identifier)...")
df.drop('Country', axis=1, inplace=True)

print(f"✅ Final feature set: {len(df.columns)} columns")

# ===============================
# IMPUTER (MEDIAN)
# ===============================
print("\nFitting SimpleImputer (median strategy)...")
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
print("✅ Imputer fitted")

# ===============================
# LOG TRANSFORMATION (OUTLIER REDUCTION)
# ===============================
print("\nApplying log transformation to skewed features...")
out_cols = ['Business Tax Rate', 'Hours to do Tax', 'Tourism Outbound',
            'Energy Usage', 'GDP', 'Health Exp/Capita',
            'Tourism Inbound', 'CO2 Emissions',
            'Days to Start Business', 'Population Total',
            'Lending Interest']

for col in out_cols:
    df_imputed[col] = np.log1p(df_imputed[col])

print(f"✅ Log transformation applied to {len(out_cols)} features")

# ===============================
# SAVE FEATURE NAMES
# ===============================
features = df_imputed.columns.tolist()
print(f"\n📋 Feature list: {features}")

# ===============================
# SCALER
# ===============================
print("\nFitting StandardScaler...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_imputed)
print("✅ Scaler fitted")

# ===============================
# CREATE MODEL DIRECTORY
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"\n📁 Created directory: {MODEL_DIR}/")

# ===============================
# SAVE PICKLE FILES
# ===============================
print("\nSaving preprocessor objects...")

with open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb") as f:
    pickle.dump(imputer, f)
print("✅ imputer.pkl saved")

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl saved")

with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f:
    pickle.dump(features, f)
print("✅ features.pkl saved")

print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
print("="*50)
print(f"Files saved in '{MODEL_DIR}/' directory:")
print("  - imputer.pkl")
print("  - scaler.pkl")
print("  - features.pkl")
print("\n⚠️  IMPORTANT: Upload the entire 'model/' folder to GitHub")
print("="*50)
