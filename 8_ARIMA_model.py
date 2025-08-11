import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import json
import warnings
warnings.filterwarnings("ignore")

# Set random seed
RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

TRAIN_TEST_SPLIT_RATIO = 1.0  

# # cs2
# FILE_PATH = 'cs2/CryoSat-2_marked.csv'
# FEATURES = ['Brouwer mean motion'] 
# FEATURE_WEIGHTS = [0.4779]  
# OUTPUT_DIR = 'cs2_ARIMA'
# PLOT_START_DATE = '2013-01-01'
# PLOT_END_DATE = '2013-12-31'

# # fy2f
# FILE_PATH = 'fy2f/Fengyun-2F_marked.csv'
# FEATURES = ['Brouwer mean motion', 'eccentricity']
# FEATURE_WEIGHTS = [0.4262, 0.3294]  
# OUTPUT_DIR = 'fy2f_ARIMA'
# PLOT_START_DATE = '2014-01-01'
# PLOT_END_DATE = '2014-12-31'

# s3a
FILE_PATH = 's3a/Sentinel-3A_marked.csv'
FEATURES = ['Brouwer mean motion']
FEATURE_WEIGHTS = [0.6139]
OUTPUT_DIR = 's3a_ARIMA'
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
        ax.plot(plot_dates, predictions[i][mask], label='ARIMA Predictions')
        ax.set_title(f'{feature} - True vs Predicted Values', fontsize=22)
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
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 3 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    for i, (feature, ax) in enumerate(zip(features, axes)):
        ax.plot(plot_dates, residuals[i][mask], label='ARIMA Residuals')
        
        labels = True
        for maneuver_date in maneuver_dates:
            if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
                if labels:
                    ax.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5, label='True Maneuvers')
                    labels = False
                else:
                    ax.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{feature} - ARIMA Residuals', fontsize=22)
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
    # Plot combined weighted residuals with threshold.
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    
    plt.figure(figsize=(7, 4))
    plt.plot(dates[mask], residuals[mask], label='Weighted Combined Residuals')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Best Threshold: {threshold:.3f}')
    
    labels = True
    for maneuver_date in maneuver_dates:
        if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
            if labels:
                plt.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5, label='True Maneuvers')
                labels = False
            else:
                plt.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5)
    
    plt.title('Weighted Combined Normalized Residuals', fontsize=20)
    plt.ylabel('Combined Residual Value', fontsize=17)
    plt.legend(fontsize=13, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=14)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    interval = max(1, min(500, len(dates[mask]) // 6))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    plt.gcf().autofmt_xdate()
    
    plt.savefig(os.path.join(output_dir, 'combined_residuals.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, output_dir):
    # Plot threshold search results.
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

if __name__ == "__main__":
    data = pd.read_csv(FILE_PATH)
    data['time'] = pd.to_datetime(data['time']).dt.normalize()
    dates = data['time']
    maneuver_dates = dates[data['is_maneuver'] == 1].tolist()
    
    all_residuals, all_true_values, all_predictions = [], [], []
    arima_models = {}
    
    for feature in FEATURES:
        print(f"Processing feature: {feature}")
        
        # Load ARIMA parameters
        param_file = os.path.join(OUTPUT_DIR, f'{feature}_arima_params.json')
        with open(param_file, 'r') as f:
            params = json.load(f)
        p_range = range(params['p_range'][0], params['p_range'][1] + 1)
        d_range = range(params['d_range'][0], params['d_range'][1] + 1)
        q_range = range(params['q_range'][0], params['q_range'][1] + 1)
        
        # Standardize data
        scaler = StandardScaler()
        feature_data = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        
        best_f1 = 0
        best_model = None
        best_params = None
        
        # Grid search using F1 score
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    model = ARIMA(feature_data, order=(p, d, q))
                    model = model.fit()
                    
                    feature_predictions = model.fittedvalues
                    residuals = np.abs(feature_data - feature_predictions)
                    
                    thresholds_low = np.linspace(np.percentile(residuals, 0), 
                            np.percentile(residuals, 50), 10)
                    thresholds_high = np.linspace(np.percentile(residuals, 99.5), 
                                np.percentile(residuals, 99.9), 10)
                    thresholds_mid = np.linspace(np.percentile(residuals, 50), 
                            np.percentile(residuals, 99.5), 20)
                    thresholds = np.concatenate([thresholds_low, thresholds_mid, thresholds_high])
                    
                    f1_scores = []
                    for threshold in thresholds:
                        predicted = residuals > threshold
                        f1_scores.append(f1_score(data['is_maneuver'].values, predicted))
                    max_f1 = max(f1_scores)
                    
                    if max_f1 > best_f1:
                        best_f1 = max_f1
                        best_params = (p, d, q)
                        best_model = model
                    print(f'Completed combination {[p,d,q]}, F1={max_f1:.3f}')
            
        print(f"Best ARIMA parameters - p: {best_params[0]}, d: {best_params[1]}, q: {best_params[2]}, F1: {best_f1:.3f}")
        
        # Use best model for prediction
        feature_predictions = best_model.fittedvalues
        residuals = np.abs(feature_data - feature_predictions)
        
        all_residuals.append(residuals)
        arima_models[feature] = best_model
        all_true_values.append(feature_data)
        all_predictions.append(feature_predictions)
    
    # Plot feature comparison
    plot_feature_comparison(FEATURES, all_true_values, all_predictions, all_residuals,
                           dates, maneuver_dates, PLOT_START_DATE, PLOT_END_DATE, OUTPUT_DIR)
    
    # Combine residuals with weights
    normalized_residuals = [MinMaxScaler().fit_transform(r.reshape(-1, 1)).flatten() 
                           for r in all_residuals]
    weighted_residuals = np.sqrt(sum(w * (r ** 2) for r, w in zip(normalized_residuals, FEATURE_WEIGHTS)))
    
    # Find optimal threshold
    thresholds_low = np.linspace(np.percentile(weighted_residuals, 0), 
                                np.percentile(weighted_residuals, 50), 10)
    thresholds_high = np.linspace(np.percentile(weighted_residuals, 99.5), 
                                 np.percentile(weighted_residuals, 99.9), 10)
    thresholds_mid = np.linspace(np.percentile(weighted_residuals, 50), 
                                np.percentile(weighted_residuals, 99.5), 20)
    thresholds = np.concatenate([thresholds_low, thresholds_mid, thresholds_high])
    
    f1_scores, precisions, recalls = [], [], []
    for threshold in thresholds:
        predicted = weighted_residuals > threshold
        f1_scores.append(f1_score(data['is_maneuver'].values, predicted))
        precisions.append(precision_score(data['is_maneuver'].values, predicted, zero_division=0))
        recalls.append(recall_score(data['is_maneuver'].values, predicted, zero_division=0))
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Save precision and recall values
    pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precisions, 'recall': recalls})
    pr_df.to_csv(os.path.join(OUTPUT_DIR, 'PRdata.csv'), index=False)
    
    # Generate plots
    plot_combined_residuals(weighted_residuals, best_threshold, dates, maneuver_dates,
                           PLOT_START_DATE, PLOT_END_DATE, OUTPUT_DIR)
    plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, OUTPUT_DIR)
    
    # Save model info
    model_info = {}
    for feature, model in arima_models.items():
        params_dict = pd.Series(model.params, index=model.param_names).to_dict()
        
        model_info[feature] = {
            'order': model.model.order,
            'aic': model.aic,
            'bic': model.bic,
            'params': params_dict
        }
    
    with open(os.path.join(OUTPUT_DIR, 'arima_model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {max(f1_scores):.3f}")