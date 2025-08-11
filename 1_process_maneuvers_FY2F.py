# Extract start time and end time of maneuvers from the FY2F maneuver file
import pandas as pd
from datetime import datetime
import re

def parse_time(time_str):
    """
    Convert time string to datetime object
    """
    time_str = time_str.replace('"', '').replace(' CST', '')
    time_str = time_str.replace('T', ' ')
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

def abstract_maneuver_file(input_file, output_file):
    maneuver_times = []
    
    with open(input_file, 'r') as f:
        for line in f:
            matches = re.findall(r'"([^"]+)"', line)
            
            # Extract start time and end time from the line
            start_time_str = matches[0]
            end_time_str = matches[1]
            
            start_time = parse_time(start_time_str)
            end_time = parse_time(end_time_str)
            
            maneuver_times.append({
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(maneuver_times)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    abstract_maneuver_file('fy2f/manFY2F.txt.fy', 'fy2f/fy2f_maneuver_times.csv') 