"""
Analyze orbital data features and plot
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_orbital_features(input_file, name, output_dir='figure', start_date=None, end_date=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(input_file)
    df_copy = df.copy()
    
    if start_date is not None:
        df = df[df.iloc[:, 0] >= start_date]
    if end_date is not None:
        df = df[df.iloc[:, 0] <= end_date]

    features = ['eccentricity', 'argument of perigee', 'inclination', 
                'mean anomaly', 'Brouwer mean motion', 'right ascension']
    
    corr_matrix = df_copy[features].corr(method='pearson')
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix, 
                annot=True,  
                cmap='coolwarm', 
                vmin=-1, vmax=1,
                center=0,  
                square=True,  
                fmt='.3f',  
                annot_kws={"size": 19})  
    plt.title(f'Feature Correlations in {name} Data', fontsize=22)
    plt.xticks(fontsize=19, rotation=30, ha='right')
    plt.yticks(fontsize=19)
    plt.tight_layout()
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=19)  
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=1080, bbox_inches='tight')
    plt.close()
    
    # Plot all features time series
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    axes = axes.flatten() 
    
    time_data = pd.to_datetime(df.iloc[:, 0])
    
    for idx, (feature, ax) in enumerate(zip(features, axes)):
        ax.plot(time_data, df[feature], linewidth=1)
        ax.set_title(feature, fontsize=22)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel(feature, fontsize=20)
        ax.tick_params(axis='x', rotation=30, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.offsetText.set_fontsize(16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_features_time_series.png', dpi=1080, bbox_inches='tight')
    plt.close()
    
    # Plot features diff with maneuvers
    fig, axes = plt.subplots(3, 2, figsize=(20, 13))
    axes = axes.flatten()
    
    maneuver_points = time_data[df['is_maneuver'] == 1]
    
    for idx, (feature, ax) in enumerate(zip(features, axes)):
        diff_values = df[feature].diff()
        ax.plot(time_data, diff_values, linewidth=1, label='Feature diff')
        
        for mtime in maneuver_points:
            ax.axvline(x=mtime, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{feature} (diff) with Maneuvers', fontsize=22)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel(f'Δ{feature}', fontsize=20)
        ax.tick_params(axis='x', rotation=30, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.offsetText.set_fontsize(16)
    axes[0].legend(['Differenced feature series', 'Maneuver point'], fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/features_diff_with_maneuvers.png', dpi=1080, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Sentinel-3
    analyze_orbital_features(
        input_file='s3a/Sentinel-3A_marked.csv',
        name = 'Sentinel-3A',
        output_dir='s3a_figure',
        start_date = '2018-01-01',
        end_date = '2019-12-31'
    )
    # Fengyun-2F
    analyze_orbital_features(
        input_file='fy2f/Fengyun-2F_marked.csv',
        name = 'Fengyun-2F',
        output_dir='fy2f_figure',
        start_date = '2017-01-01',
        end_date = '2017-12-31'
    )
    # CryoSat-2
    analyze_orbital_features(
        input_file='cs2/CryoSat-2_marked.csv',
        name = 'CryoSat-2',
        output_dir='cs2_figure',
        start_date = '2013-01-01',
        end_date = '2013-12-31'
    )
