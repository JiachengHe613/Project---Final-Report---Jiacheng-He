"""Calculate F1 scores for satellite orbital features using a 3-day EMA model"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from scipy import stats
import os
import numpy as np

SATELLITES = {
    'Sentinel-3A': 's3a/Sentinel-3A_marked.csv',
    'Fengyun-2F': 'fy2f/Fengyun-2F_marked.csv',
    'CryoSat-2': 'cs2/CryoSat-2_marked.csv'
}

FEATURES = [
    'eccentricity', 'argument of perigee', 'inclination',
    'mean anomaly', 'Brouwer mean motion', 'right ascension'
]

def calculate_f1_score(df, feature, ema_window=3):
    """Calculate the optimal F1 score for a single feature."""
    # Calculate EMA deviation
    ema = df[feature].ewm(span=ema_window, adjust=False).mean()
    deviation = abs(df[feature] - ema)
    valid_mask = ~deviation.isna()
    
    if not valid_mask.any() or not df['is_maneuver'][valid_mask].any():
        return 0
        
    # Standardize and handle outliers
    z_scores = stats.zscore(deviation[valid_mask], nan_policy='omit')
    z_scores = abs(pd.Series(z_scores).fillna(0))
    
    # Set threshold range
    min_threshold = np.percentile(z_scores, 50)
    max_threshold = np.percentile(z_scores, 99.5)
    thresholds = np.linspace(min_threshold, max_threshold, 200)
    
    # Find the optimal threshold
    best_f1 = 0
    
    for threshold in thresholds:
        predicted = z_scores > threshold
        f1 = f1_score(df['is_maneuver'][valid_mask], predicted, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
    
    return best_f1

def evaluate_features(data_file):
    """Calculate F1 scores for all features."""
    df = pd.read_csv(data_file)
    
    if 'is_maneuver' not in df.columns:
        return {f: 0 for f in FEATURES}
        
    df['time'] = pd.to_datetime(df['time'])
    results = {}
    
    for feature in FEATURES:
        if feature in df.columns:
            f1 = calculate_f1_score(df, feature)
            results[feature] = f1
        else:
            results[feature] = 0
            
    return results

def plot_heatmap(results):
    # Convert data to DataFrame and plot heatmap
    plt.figure(figsize=(10, 6))
    
    results_df = pd.DataFrame(results).T
    ax = sns.heatmap(
        results_df,
        annot=True,
        cmap='YlGnBu',
        fmt='.4f',
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'F1 score'},
        annot_kws={"size": 16}
    )
    
    plt.title('The F1 score of 3-day EMA model for each satellite feature', fontsize=18)
    plt.xticks(fontsize=14, rotation=35, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('F1 score', fontsize=14)

    ax.collections[0].colorbar.ax.tick_params(labelsize=15)
    os.makedirs('combo_figure', exist_ok=True)
    plt.savefig('combo_figure/features_3day_ema_f1_heatmap.png', dpi=1080, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    all_results = {}
    for sat_name, data_file in SATELLITES.items():
        all_results[sat_name] = evaluate_features(data_file)
    
    plot_heatmap(all_results) 