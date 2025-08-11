"""Process satellite data: check for missing values, interpolate, and mark maneuver points"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def process_satellite_data(file_path, maneuver_file=None, start_date=None, end_date=None, output_file=None):
    df = pd.read_csv(file_path)
    df.columns = ['time'] + list(df.columns[1:])
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df = df.sort_values('time')
    
    # Filter by time range
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date).date()]

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values:", missing[missing > 0])
    
    # Handle missing dates
    date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    date_range = pd.Series(date_range.date)
    missing_dates = set(date_range) - set(df['date'])
    
    if missing_dates:
        df.set_index('date', inplace=True)
        
        for date in sorted(missing_dates):
            prev_records = df.loc[df.index < date]
            prev_time = prev_records['time'].iloc[-1].time() if not prev_records.empty else datetime.time(4, 0)
            df.loc[date, 'time'] = datetime.combine(date, prev_time)
        
        # Cubic polynomial interpolation
        for date in sorted(missing_dates):
            window = pd.Timedelta(days=25)  # Use data 25 days before and after for interpolation
            mask = (df.index >= date - window) & (df.index <= date + window)
            
            for col in df.columns:
                if col == 'time':
                    continue
                valid_data = df.loc[mask, col].dropna()
                if len(valid_data) >= 4:  # Requires at least 4 points for cubic polynomial fitting
                    x = np.array([(d - date).days for d in valid_data.index])
                    df.loc[date, col] = np.poly1d(np.polyfit(x, valid_data.values, 3))(0)
        
        df = df.sort_index().reset_index(drop=True)
    else:
        df = df.drop('date', axis=1)
    
    df = df.sort_values('time').reset_index(drop=True)
    
    # Mark maneuver time points
    if maneuver_file and os.path.exists(maneuver_file):
        maneuvers = pd.read_csv(maneuver_file)
        maneuvers['start_time'] = pd.to_datetime(maneuvers['start_time'])
        maneuvers['end_time'] = pd.to_datetime(maneuvers['end_time'])
        maneuvers = maneuvers.sort_values('start_time')
        
        df['is_maneuver'] = False
        data_range = (df['time'].min() - timedelta(days=1), df['time'].max())
        valid_maneuvers = maneuvers[
            (maneuvers['start_time'] >= data_range[0]) & 
            (maneuvers['start_time'] <= data_range[1])
        ]
        
        for _, man in valid_maneuvers.iterrows():
            # Mark during maneuver and the first point after its end
            mask = (df['time'] >= man['start_time']) & (df['time'] <= man['end_time'])
            df.loc[mask, 'is_maneuver'] = True
            
            next_points = df[df['time'] > man['end_time']]
            if not next_points.empty:
                df.loc[next_points.index[0], 'is_maneuver'] = True
        
        print(f"Marked {df['is_maneuver'].sum()}/{len(df)} maneuver points")
    else:
        df['is_maneuver'] = False
    
    # Save results
    if not output_file:
        path = Path(file_path)
        output_file = str(path.parent / f"{path.stem}_marked{path.suffix}")
    
    df.to_csv(output_file, index=False)
    return output_file

if __name__ == "__main__":
    satellites = {
        'Sentinel-3A': {
            'data': 's3a/Sentinel-3A.csv',
            'maneuvers': 's3a/s3a_maneuver_times.csv',
            'output': 's3a/Sentinel-3A_marked.csv',
            #'start_date': '2016-07-01'
        },
        'Fengyun-2F': {
            'data': 'fy2f/Fengyun-2F.csv',
            'maneuvers': 'fy2f/fy2f_maneuver_times.csv',
            'output': 'fy2f/Fengyun-2F_marked.csv'
        },
        'CryoSat-2': {
            'data': 'cs2/CryoSat-2.csv',
            'maneuvers': 'cs2/cs2_maneuver_times.csv',
            'output': 'cs2/CryoSat-2_marked.csv'
        }
    }
    
    for name, config in satellites.items():
        print(f"\nProcessing {name}...")
        process_satellite_data(
            file_path=config['data'],
            maneuver_file=config['maneuvers'],
            output_file=config['output'],
            start_date=config.get('start_date'),
            end_date=config.get('end_date')
        ) 