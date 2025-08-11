import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

EMA_WINDOW = 3

# # cs2
# FILE_PATH = 'cs2/CryoSat-2_marked.csv'
# FEATURES = ['Brouwer mean motion'] 
# FEATURE_WEIGHTS = [0.4779]  
# OUTPUT_DIR = 'cs2_EMA'
# PLOT_START_DATE = '2013-01-01'
# PLOT_END_DATE = '2013-12-31'

# # fy2f
# FILE_PATH = 'fy2f/Fengyun-2F_marked.csv'
# FEATURES = ['Brouwer mean motion', 'eccentricity']
# FEATURE_WEIGHTS = [0.4262, 0.3294]  
# OUTPUT_DIR = 'fy2f_EMA'
# PLOT_START_DATE = '2014-01-01'
# PLOT_END_DATE = '2014-12-31'

# s3a
FILE_PATH = 's3a/Sentinel-3A_marked.csv'
FEATURES = ['Brouwer mean motion']
FEATURE_WEIGHTS = [0.6139]
OUTPUT_DIR = 's3a_EMA'
PLOT_START_DATE = '2017-01-01'
PLOT_END_DATE = '2017-12-31'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_feature_comparison(features, true_values, predictions, residuals, dates, 
                          maneuver_dates, start_date, end_date, output_dir):
    """Plot true vs predicted values and residuals for all features."""
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    plot_dates = dates[mask]
    
    # Predictions plot
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 3 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    for i, (feature, ax) in enumerate(zip(features, axes)):
        ax.plot(plot_dates, true_values[i][mask], label='True Values')
        ax.plot(plot_dates, predictions[i][mask], label='EMA Predictions')
        ax.set_title(f'{feature} - True vs EMA Values', fontsize=22)
        ax.set_ylabel('Value', fontsize=20)
        ax.legend(fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=16)
        ax.tick_params(axis='y', which='major', labelsize=14)
        ax.yaxis.offsetText.set_fontsize(14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_predictions.png'), dpi=1080, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 4 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    for i, (feature, ax) in enumerate(zip(features, axes)):
        ax.plot(plot_dates, residuals[i][mask], label='Residuals')
        
        labels = True
        for maneuver_date in maneuver_dates:
            if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
                if labels:
                    ax.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5, label='True Maneuvers')
                    labels = False
                else:
                    ax.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5)

        ax.set_title(f'{feature} - Residuals', fontsize=22)
        ax.set_ylabel('Residual Value', fontsize=20)
        ax.legend(fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=16)
        ax.tick_params(axis='y', which='major', labelsize=14)
        ax.yaxis.offsetText.set_fontsize(14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_residuals.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_combined_residuals(residuals, threshold, dates, maneuver_dates, 
                           start_date, end_date, output_dir):
    """Plot combined weighted residuals with threshold."""
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    
    plt.figure(figsize=(7, 4))
    plt.plot(dates[mask], residuals[mask], label='Weighted Combined Residuals')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Best Threshold: {threshold:.3f}')
    
    ax = plt.gca()
    labels = True
    for maneuver_date in maneuver_dates:
        if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
            if labels:
                ax.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5, label='True Maneuvers')
                labels = False
            else:
                ax.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5)
    
    plt.title('Weighted Combined Normalized Residuals', fontsize=20)
    plt.ylabel('Combined Residual Value', fontsize=17)
    plt.legend(fontsize=13, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=14)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    interval = max(1, min(500, len(dates[mask]) // 6))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    plt.gcf().autofmt_xdate()
    
    plt.savefig(os.path.join(output_dir, 'combined_residuals.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, output_dir):
    """Plot threshold search results."""
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, precisions, 'g-', label='Precision')
    plt.plot(thresholds, recalls, 'b-', label='Recall')
    plt.axvline(x=best_threshold, color='k', linestyle='--', 
                label=f'Best Threshold: {best_threshold:.3f}')
    
    max_f1_idx = np.argmax(f1_scores)
    plt.scatter([thresholds[max_f1_idx]], [f1_scores[max_f1_idx]], 
                color='red', label=f'Max F1: {f1_scores[max_f1_idx]:.3f}')
    
    plt.title('Metrics vs Thresholds', fontsize=16)
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    plt.savefig(os.path.join(output_dir, 'threshold_search.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def main():
    data = pd.read_csv(FILE_PATH)
    data['time'] = pd.to_datetime(data['time']).dt.normalize()
    dates = data['time']
    maneuver_dates = dates[data['is_maneuver'] == 1].tolist()
    
    all_residuals, all_true_values, all_predictions = [], [], []
    
    # Process features
    for feature in FEATURES:
        feature_data = data[feature].values
        ema_predictions = pd.Series(feature_data).ewm(span=EMA_WINDOW, adjust=False).mean()
        
        # Calculate residuals
        true_vals = feature_data[EMA_WINDOW:]
        pred_vals = ema_predictions.shift(1).iloc[EMA_WINDOW:].values
        residuals = np.abs(true_vals - pred_vals)
        
        all_true_values.append(true_vals)
        all_predictions.append(pred_vals)
        all_residuals.append(residuals)
    
    filtered_dates = dates.iloc[EMA_WINDOW:EMA_WINDOW+len(all_residuals[0])]
    
    # Plot individual features
    plot_feature_comparison(FEATURES, all_true_values, all_predictions, all_residuals,
                           filtered_dates, maneuver_dates, PLOT_START_DATE, PLOT_END_DATE, OUTPUT_DIR)
    
    # Combine residuals with weights
    normalized_residuals = [MinMaxScaler().fit_transform(r.reshape(-1, 1)).flatten() 
                           for r in all_residuals]
    weighted_residuals = np.sqrt(sum(w * (r ** 2) for r, w in zip(normalized_residuals, FEATURE_WEIGHTS)))
    
    # Find optimal threshold
    filtered_maneuvers = data['is_maneuver'].iloc[EMA_WINDOW:EMA_WINDOW+len(weighted_residuals)].values
    
    thresholds_low = np.linspace(np.percentile(weighted_residuals, 0), 
                                np.percentile(weighted_residuals, 50), 10)
    thresholds_high = np.linspace(np.percentile(weighted_residuals, 99.5), 
                                 np.percentile(weighted_residuals, 99.9), 10)
    thresholds_mid = np.linspace(np.percentile(weighted_residuals, 50), 
                                np.percentile(weighted_residuals, 99.5), 20)
    thresholds = np.concatenate([thresholds_low, thresholds_mid, thresholds_high])
    
    metrics = {'f1': [], 'precision': [], 'recall': []}
    for threshold in thresholds:
        predicted = weighted_residuals > threshold
        metrics['f1'].append(f1_score(filtered_maneuvers, predicted))
        metrics['precision'].append(precision_score(filtered_maneuvers, predicted, zero_division=0))
        metrics['recall'].append(recall_score(filtered_maneuvers, predicted, zero_division=0))
    f1_scores, precisions, recalls = metrics['f1'], metrics['precision'], metrics['recall']

    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Save precision and recall values
    pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precisions, 'recall': recalls})
    pr_df.to_csv(os.path.join(OUTPUT_DIR, 'PRdata.csv'), index=False)
    
    # Generate plots
    plot_combined_residuals(weighted_residuals, best_threshold, filtered_dates, 
                           maneuver_dates, PLOT_START_DATE, PLOT_END_DATE, OUTPUT_DIR)
    plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, OUTPUT_DIR)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {max(f1_scores):.3f}")

if __name__ == "__main__":
    main() 