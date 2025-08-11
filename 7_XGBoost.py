import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import random
import matplotlib.dates as mdates

# Set random seed
RANDOM_SEED = 666
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# set constants
TRAIN_TEST_SPLIT_RATIO = 1.0
XGB_PARAMS = {
    'learning_rate': 0.05,
    'eval_metric': 'mae',
    'early_stopping_rounds': 10
}
N_ESTIMATORS_RANGE = [30,50,100,150,200,300] 
MAX_DEPTH_RANGE = [2,3,5,7] 


# # cs2
# FILE_PATH = 'cs2/CryoSat-2_marked.csv'
# FEATURES = ['Brouwer mean motion'] 
# FEATURE_WEIGHTS = [0.4779]  
# NUM_LAGS = 70#50 
# OUTPUT_DIR = 'cs2_XGBoost'
# PLOT_START_DATE = '2013-01-01'
# PLOT_END_DATE = '2013-12-31'


# # fy2f
# FILE_PATH = 'fy2f/Fengyun-2F_marked.csv'
# FEATURES = ['Brouwer mean motion', 'eccentricity']
# FEATURE_WEIGHTS = [0.4262, 0.3294]  
# NUM_LAGS = 70#40
# OUTPUT_DIR = 'fy2f_XGBoost'
# PLOT_START_DATE = '2014-01-01'
# PLOT_END_DATE = '2014-12-31'


# s3a
FILE_PATH = 's3a/Sentinel-3A_marked.csv'
FEATURES = ['Brouwer mean motion']
FEATURE_WEIGHTS = [0.6139]
NUM_LAGS = 125#125
OUTPUT_DIR = 's3a_XGBoost'
PLOT_START_DATE = '2017-01-01'
PLOT_END_DATE = '2017-12-31'


os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_xgb_model(X_train, y_train, n_estimators, max_depth):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        **XGB_PARAMS
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    return model

def plot_results(features, all_true_values, all_predictions, all_residuals, dates, maneuver_dates, 
                start_date, end_date):
    # Filter data for plotting
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    plot_dates = dates[mask]
    
    num_features = len(features)
    # Plot prediction comparison for all features
    plt.figure(figsize=(10, 3 * num_features))
    for i, feature in enumerate(features):
        ax = plt.subplot(num_features, 1, i+1)
        plt.plot(plot_dates, all_true_values[i][mask], label='True Values')
        plt.plot(plot_dates, all_predictions[i][mask], label='Predictions')
        plt.title(f'{feature} - True vs Predicted Values', fontsize=22)
        plt.legend(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_ylabel('Value', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_predictions.png'), dpi=1080, bbox_inches='tight')
    plt.close()
    
    # Plot residuals
    plt.figure(figsize=(10, 3 * num_features))
    for i, feature in enumerate(features):
        ax = plt.subplot(num_features, 1, i+1)
        plt.plot(plot_dates, all_residuals[i][mask], label='Residuals')
        
        labels = True
        for maneuver_date in maneuver_dates:
            if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
                if labels:
                    plt.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5, label='True Maneuvers')
                    labels = False
                else:
                    plt.axvline(x=maneuver_date, color='r', linestyle='--', alpha=0.5)
                    
        plt.title(f'{feature} - Residuals', fontsize=22)
        plt.legend(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_ylabel('Residual Value', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_residuals.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_combined_residuals(weighted_residuals, best_threshold, dates, maneuver_dates, 
                           start_date, end_date):
    # Filter data for plotting
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    plot_dates = dates[mask]
    plot_residuals = weighted_residuals[mask]
    
    plt.figure(figsize=(7, 5))
    plt.plot(plot_dates, plot_residuals, label='Weighted Combined Residuals')
    plt.axhline(y=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    
    labels = True
    for maneuver_date in maneuver_dates:
        if start_date <= maneuver_date.strftime('%Y-%m-%d') <= end_date:
            if labels:
                plt.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5, label='True Maneuvers')
                labels = False
            else:
                plt.axvline(x=maneuver_date, color='g', linestyle='--', alpha=0.5)

    plt.title('Weighted Combined Normalized Residuals', fontsize=20)
    plt.legend(fontsize=13)#, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7) 
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(date_format)
    interval = max(1, min(500, len(plot_dates) // 6))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=14)
    plt.ylabel('Combined Residual Value', fontsize=17)
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_residuals.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, output_dir):
    """Plot threshold search results."""
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, precisions, 'g-', label='Precision')
    plt.plot(thresholds, recalls, 'b-', label='Recall')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    max_f1 = max(f1_scores)
    max_f1_idx = np.argmax(f1_scores)
    plt.scatter([thresholds[max_f1_idx]], [max_f1], color='red', label=f'Max F1: {max_f1:.3f}')
    plt.title('Metrics vs Thresholds', fontsize=16)
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(output_dir, 'threshold_search.png'), dpi=1080, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_importances, output_dir):
    # Plot feature importance chart for XGBoost
    all_importance = {}
    for feature, importances in feature_importances.items():
        for i, imp in enumerate(importances):
            all_importance[f"{feature}_lag_{i+1}"] = imp
    
    feature_importance_dict = {}
    for key, value in all_importance.items():
        parts = key.split('_lag_')
        feature_name = parts[0]
        lag_num = int(parts[1])
        if feature_name not in feature_importance_dict:
            feature_importance_dict[feature_name] = []
        feature_importance_dict[feature_name].append([lag_num, value])
    
    for feature_name, importance_list in feature_importance_dict.items():
        importance_list.sort(key=lambda x: x[0])
        feature_df = pd.DataFrame(importance_list, columns=['Lag', 'Importance'])
        feature_df.to_csv(os.path.join(output_dir, f'{feature_name}_importance.csv'), index=False)
    
    # Plot importance
    num_features = len(feature_importance_dict)
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features))
    if num_features == 1:
        axes = [axes]
    
    for i, (feature_name, importance_list) in enumerate(feature_importance_dict.items()):
        lags = [item[0] for item in importance_list]
        importances = [item[1] for item in importance_list]
        axes[i].bar(lags, importances)
        axes[i].set_xticks(range(0, max(lags) + 1, 10))
        axes[i].set_xticklabels([f'Lag {j}' for j in range(0, max(lags) + 1, 10)], fontsize=20, rotation=-20, ha='left')
        axes[i].set_title(f'{feature_name} - Feature Importance by Lag', fontsize=22)
        axes[i].set_xlabel('Lag', fontsize=20)
        axes[i].set_ylabel('Importance', fontsize=20)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_combined.png'), dpi=1080, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    data = pd.read_csv(FILE_PATH)
    
    # Convert time data
    data['time'] = pd.to_datetime(data['time']).dt.normalize()
    dates = data['time']
    
    # Get indices and dates of true maneuvers
    maneuver_indices = np.where(data['is_maneuver'] == 1)[0]
    maneuver_dates = dates[maneuver_indices].tolist()
    
    all_residuals = []
    feature_importances = {}
    all_true_values = [] 
    all_predictions = []  
    
    for i, feature in enumerate(FEATURES):
        print(f"Processing feature: {feature}")
        
        # Standardize data
        scaler = StandardScaler()
        feature_data = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        
        df_feature = pd.DataFrame({feature: feature_data})
        
        lagged_dfs = []
        for lag in range(1, NUM_LAGS + 1):
            for col in df_feature.columns:
                lagged_df = df_feature[col].shift(lag).to_frame(f'{col}_lag_{lag}')
                lagged_dfs.append(lagged_df)
        
        X = pd.concat(lagged_dfs, axis=1)
        X = X.iloc[NUM_LAGS:]
        y = df_feature.iloc[NUM_LAGS:]

        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Grid search for best parameters
        for n_estimators in N_ESTIMATORS_RANGE:
            for max_depth in MAX_DEPTH_RANGE:
                model = train_xgb_model(X, y[feature], n_estimators, max_depth)
                feature_predictions = model.predict(X)
                residuals = np.abs(y[feature].values - feature_predictions)
                
                # Calculate F1 score
                aligned_maneuvers = data['is_maneuver'].iloc[NUM_LAGS:NUM_LAGS+len(residuals)].values

                thresholds_low = np.linspace(np.percentile(residuals, 0), 
                                            np.percentile(residuals, 50), 10)
                thresholds_high = np.linspace(np.percentile(residuals, 99.5), 
                                            np.percentile(residuals, 99.9), 10)
                thresholds_mid = np.linspace(np.percentile(residuals, 50), 
                                            np.percentile(residuals, 99.5), 20)
                thresholds = np.concatenate([thresholds_low, thresholds_mid, thresholds_high])

                f1_scores = []
                for threshold in thresholds:
                    predicted_maneuvers = residuals > threshold
                    f1 = f1_score(aligned_maneuvers, predicted_maneuvers)
                    f1_scores.append(f1)
                
                max_f1 = max(f1_scores) if f1_scores else 0
                
                if max_f1 > best_score:
                    best_score = max_f1
                    best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                    best_model = model
        
        print(f"Best parameters - n_estimators: {best_params['n_estimators']}, max_depth: {best_params['max_depth']}")
        
        # Use the best model for prediction
        feature_predictions = best_model.predict(X)
        residuals = np.abs(y[feature].values - feature_predictions)
        
        all_residuals.append(residuals)
        feature_importances[feature] = best_model.feature_importances_
        all_true_values.append(y[feature].values)
        all_predictions.append(feature_predictions)
    
    filtered_dates = dates.iloc[NUM_LAGS:NUM_LAGS+len(all_residuals[0])]
    
    plot_results(FEATURES, all_true_values, all_predictions, all_residuals, 
                filtered_dates, maneuver_dates, PLOT_START_DATE, PLOT_END_DATE)
    
    normalized_residuals = []
    for residuals in all_residuals:
        normalized = MinMaxScaler().fit_transform(residuals.reshape(-1, 1)).flatten()
        normalized_residuals.append(normalized)
    
    # Calculate weighted combined residuals
    weighted_residuals = np.sqrt(sum(weight * (residual ** 2) for residual, weight in zip(normalized_residuals, FEATURE_WEIGHTS)))
    
    filtered_maneuvers = data['is_maneuver'].iloc[NUM_LAGS:NUM_LAGS+len(weighted_residuals)].values
    
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
        predicted_maneuvers = weighted_residuals > threshold
        f1_scores.append(f1_score(filtered_maneuvers, predicted_maneuvers))
        precisions.append(precision_score(filtered_maneuvers, predicted_maneuvers, zero_division=0))
        recalls.append(recall_score(filtered_maneuvers, predicted_maneuvers, zero_division=0))

    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Save precision and recall values
    pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precisions, 'recall': recalls})
    pr_df.to_csv(os.path.join(OUTPUT_DIR, 'PRdata.csv'), index=False)
    
    # Generate plots
    plot_combined_residuals(weighted_residuals, best_threshold, filtered_dates, maneuver_dates,
                           PLOT_START_DATE, PLOT_END_DATE)
    
    plot_threshold_search(thresholds, f1_scores, precisions, recalls, best_threshold, OUTPUT_DIR)
    
    plot_feature_importance(feature_importances, OUTPUT_DIR)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {max_f1:.3f}")
