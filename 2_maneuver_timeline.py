"""Draw satellite maneuvering timeline graph"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path

def create_timeline_plot(maneuver_data, orbit_data, date_range=None, output_dir='combo_figure'):
    plt.figure(figsize=(15, 6))
    satellites = list(maneuver_data.keys())
    colors = {'maneuver': '#FF6B6B', 'orbit': '#90EE90'}
    
    for idx, sat in enumerate(satellites):
        # Plot Orbit Time Range
        plt.fill_between(
            [orbit_data[sat]['first_record'], orbit_data[sat]['last_record']], 
            [idx-0.3]*2, [idx+0.3]*2,
            color=colors['orbit'], alpha=0.5
        )
        
        # Plot maneuvers
        for i, row in maneuver_data[sat].iterrows():
            plt.vlines(
                row['start_time'], idx-0.3, idx+0.3, 
                colors=colors['maneuver'], linewidth=1.5
            )
    
    plt.yticks(range(len(satellites)), satellites, fontsize=18)
    plt.xticks(fontsize=18)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.title('Time range of satellite orbital data and maneuver timestamps', fontsize=22, pad=20)
    plt.xlabel('Time', fontsize=16)
    plt.xlim(date_range)
    
    legend_elements = [
        Patch(facecolor=colors['orbit'], alpha=0.5, label='Orbit data time range'),
        Line2D([0], [0], color=colors['maneuver'], lw=1.5, label='Maneuver timestamp')
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=20)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'maneuver_timeline.png', dpi=1080, bbox_inches='tight')
    plt.close()

def load_data(maneuver_file, orbit_file):
    maneuver_df = pd.read_csv(maneuver_file)
    maneuver_df['start_time'] = pd.to_datetime(maneuver_df['start_time'])
    maneuver_df['end_time'] = pd.to_datetime(maneuver_df['end_time'])
    
    orbit_df = pd.read_csv(orbit_file)
    time_col = pd.to_datetime(orbit_df.iloc[:, 0])
    return maneuver_df, {'first_record': time_col.min(), 'last_record': time_col.max()}

if __name__ == "__main__":
    satellites = {
        'Sentinel-3A': ('s3a/s3a_maneuver_times.csv', 's3a/Sentinel-3A.csv'),
        'Fengyun-2F': ('fy2f/fy2f_maneuver_times.csv', 'fy2f/Fengyun-2F.csv'),
        'CryoSat-2': ('cs2/cs2_maneuver_times.csv', 'cs2/CryoSat-2.csv')
    }
    
    maneuver_data = {}
    orbit_data = {}
    for sat, (man_file, orb_file) in satellites.items():
        maneuver_data[sat], orbit_data[sat] = load_data(man_file, orb_file)
    
    date_range = [datetime(2010, 1, 1), datetime(2022, 12, 31)]
    create_timeline_plot(maneuver_data, orbit_data, date_range)
