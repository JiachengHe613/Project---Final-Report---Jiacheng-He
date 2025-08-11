import os
import pandas as pd
import matplotlib.pyplot as plt

# Mapping satellite code in directory name to readable name
SATELLITES = {
    'cs2': 'CryoSat-2',
    'fy2f': 'Fengyun-2F',
    's3a': 'Sentinel-3A'
}

# Model directories suffixes to look for
MODELS = [
    '_EMA', '_XGBoost', '_ARIMA', '_Autoencoder', '_LSTM'
]

# Find PRdata.csv files for each satellite and model using a clear loop
pr_files = {}
for sat_code in SATELLITES.keys():
    pr_files[sat_code] = {}
    for suffix in MODELS:
        dir_name = f"{sat_code}{suffix}"
        csv_path = os.path.join(dir_name, 'PRdata.csv')
        model_name = suffix.strip('_')
        pr_files[sat_code][model_name] = csv_path

# Ensure the output directory exists
os.makedirs('combo_figure', exist_ok=True)

# Plot PR curves for each satellite
for sat_code, models in pr_files.items():
    plt.figure(figsize=(6, 5))
    for model_name, csv_path in models.items():
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='recall')
        plt.step(df['recall'], df['precision'], where='post', label=model_name)
    
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title(f"Precision-Recall Curves - {SATELLITES[sat_code]}", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(f"combo_figure/PR_curves_{sat_code}.png")
    plt.close()