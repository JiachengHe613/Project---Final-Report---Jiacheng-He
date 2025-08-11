import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# Set random seed
RANDOM_SEED = 666
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Constants
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.002
PATIENCE = 30  


# cs2
WINDOW_SIZE = 60  
FILE_PATH = 'cs2/CryoSat-2_marked.csv'
FEATURES = ['Brouwer mean motion']
FEATURE_WEIGHTS = [0.4779]
OUTPUT_DIR = 'cs2_LSTM'
PLOT_START_DATE = '2013-01-01'
PLOT_END_DATE = '2013-12-31'

# # fy2f
# WINDOW_SIZE = 40#40  
# FILE_PATH = 'fy2f/Fengyun-2F_marked.csv'
# FEATURES = ['Brouwer mean motion', 'eccentricity']
# FEATURE_WEIGHTS = [0.4262, 0.3294]
# OUTPUT_DIR = 'fy2f_LSTM'
# PLOT_START_DATE = '2014-01-01'
# PLOT_END_DATE = '2014-12-31'

# # s3a
# WINDOW_SIZE = 80#80
# FILE_PATH = 's3a/Sentinel-3A_marked.csv'
# FEATURES = ['Brouwer mean motion']
# FEATURE_WEIGHTS = [0.6139]
# OUTPUT_DIR = 's3a_LSTM'
# PLOT_START_DATE = '2017-01-01'
# PLOT_END_DATE = '2017-12-31'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM network
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() 
        )
 
    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

def plot_feature_comparison(features, true_values, predictions, residuals, dates, 
                          maneuver_dates, start_date, end_date, output_dir):
    # Plot true vs predicted values and residuals for all features.
    mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    plot_dates = dates[mask]
    
    # Predictions plot
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 3 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    for i, (feature, ax) in enumerate(zip(features, axes)):
        ax.plot(plot_dates, true_values[i][mask], label='True Values')
        ax.plot(plot_dates, predictions[i][mask], label='Predicted Values')
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

def train_LSTM(model, X_train, y_train, X_val, y_val, epochs, learning_rate, patience=20):
    """Train LSTM with early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training data
    train_tensor = torch.FloatTensor(X_train).to(device)
    train_target_tensor = torch.FloatTensor(y_train).to(device)
    train_dataset = TensorDataset(train_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation data
    val_tensor = torch.FloatTensor(X_val).to(device)
    val_target_tensor = torch.FloatTensor(y_val).to(device)
    val_dataset = TensorDataset(val_tensor, val_target_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_epochs += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if no_improve_epochs >= patience:
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': len(train_losses) - no_improve_epochs
    }

if __name__ == "__main__":
    data = pd.read_csv(FILE_PATH)
    data['time'] = pd.to_datetime(data['time']).dt.normalize()
    dates = data['time']
    maneuver_dates = dates[data['is_maneuver'] == 1].tolist()
    
    all_residuals, all_true_values, all_predicted = [], [], []
    feature_best_info = []
    
    for feature in FEATURES:
        print(f"Processing feature: {feature}")
        
        # Normalize data
        scaler = MinMaxScaler()
        feature_data = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
        
        # Create prediction windows
        X, y = [], []
        for i in range(len(feature_data) - WINDOW_SIZE):
            window = feature_data[i:i + WINDOW_SIZE]
            target = feature_data[i + WINDOW_SIZE]
            X.append(window)
            y.append(target)
        X, y = np.array(X), np.array(y)
        
        X = X.reshape(-1, 1, WINDOW_SIZE).astype(np.float32)
        y = y.reshape(-1, 1).astype(np.float32)
        
        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        model = LSTM(input_size=1)
        
        history = train_LSTM(model, X_train, y_train, X_val, y_val, EPOCHS, LEARNING_RATE, PATIENCE)
        
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            val_predictions = model(X_val_tensor)
            final_val_loss = nn.MSELoss()(val_predictions, y_val_tensor).item()
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predicted_values = model(X_tensor).cpu().numpy().flatten()
        
        # Calculate prediction errors
        prediction_errors = np.abs(y.flatten() - predicted_values)
        
        aligned_predictions = np.zeros(len(feature_data))
        aligned_predictions[WINDOW_SIZE:] = predicted_values
        aligned_predictions[:WINDOW_SIZE] = predicted_values[0]  # Fill start with first prediction
        
        aligned_errors = np.zeros(len(feature_data))
        aligned_errors[WINDOW_SIZE:] = prediction_errors
        aligned_errors[:WINDOW_SIZE] = prediction_errors[0]  # Fill start with first error
        
        all_residuals.append(aligned_errors)
        all_true_values.append(feature_data)
        all_predicted.append(aligned_predictions)
        
        feature_best_info.append({
            'feature': feature,
            'train_loss': history['train_loss'][-1],
            'val_loss': history['best_val_loss'],
            'final_val_loss': final_val_loss,
            'best_epoch': history['best_epoch']
        })

    # Plot feature comparison
    plot_feature_comparison(FEATURES, all_true_values, all_predicted, all_residuals,
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
        f1_scores.append(f1_score(data['is_maneuver'].values, predicted, zero_division=0))
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
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {max(f1_scores):.3f}") 