import pandas as pd
import sys
from datetime import datetime

# Load the data into a pandas dataframe
df = pd.read_csv(sys.argv[1])

# Convert the timestamp column to datetime type
df['time'] = pd.to_datetime(df['time'])

# Set the timestamp column as the index of the dataframe
df.set_index('time', inplace=True)

# Resample the dataframe to hourly frequency
df_resampled = df.resample('H').mean()

# Get the list of dates where the resampled data is missing
missing_dates = df_resampled[df_resampled.isna().any(axis=1)].index.strftime('%Y-%m-%d').unique().tolist()

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
program_name = "logger"
logger = open(f"{program_name}.log", 'a')
logger.write(f"\n\n\nThe program '{sys.argv[0]}' is running at {current_time}\n")
# Print the missing dates
logger.write("Dates with non-hourly data:\n")

count = 0
for date in missing_dates:
    if count % 5 == 0:
        logger.write("\n")
    logger.write(f" {str(date)} ")
    count += 1
