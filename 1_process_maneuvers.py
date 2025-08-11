# Extract start time and end time of maneuvers from the maneuver file
import pandas as pd
from datetime import datetime, timedelta

def parse_time(year, day, hour, minute):
    """Convert year, day of year, hour, minute to datetime object"""
    date = datetime(int(year), 1, 1) + timedelta(days=int(day)-1)
    return date.replace(hour=int(hour), minute=int(minute))

def abstract_maneuver_file(input_file, output_file):
    maneuver_times = []
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Extract start time
            start_year = parts[1]
            start_day = parts[2]
            start_hour = parts[3]
            start_minute = parts[4]

            end_year = parts[5]
            end_day = parts[6]
            end_hour = parts[7]
            end_minute = parts[8]
            
            # Convert to datetime objects
            start_time = parse_time(start_year, start_day, start_hour, start_minute)
            end_time = parse_time(end_year, end_day, end_hour, end_minute)
            
            maneuver_times.append({
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    df = pd.DataFrame(maneuver_times)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    abstract_maneuver_file('cs2/cs2man.txt', 'cs2/cs2_maneuver_times.csv') 
    abstract_maneuver_file('s3a/s3aman.txt', 's3a/s3a_maneuver_times.csv') 