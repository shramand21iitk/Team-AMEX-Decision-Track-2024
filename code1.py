import pandas as pd
import numpy as np

# Load the data
batsman_df = pd.read_csv('batsman_level_scorecard_with_newfeatures.csv')
bowler_df = pd.read_csv('bowler_level_scorecard_with_newfeatures.csv')
match_df = pd.read_csv('D:\AMEX\Dataset\Dataset\\train_data_with_samplefeatures.csv')

# Define the columns for the statistics we need
batsman_stats_cols = ['batting_average', 'average_strike_rate', 'average_fours_per_ball', 'average_sixes_per_ball']
bowler_stats_cols = ['average_economy', 'bowling_average', 'bowling_strike_rate', 'average_wides_per_ball', 'average_noballs_per_ball']

# Initialize new columns for the ratios
for stat in batsman_stats_cols + bowler_stats_cols:
    match_df[f'ratio_{stat}'] = 0.0

# Compute aggregated statistics and their ratios for each match
for index, row in match_df.iterrows():
    team1_ids = row['team1_roster_ids'].split(':')
    team2_ids = row['team2_roster_ids'].split(':')

    # Calculate the average statistics for team 1 and team 2
    team1_batsman_avg = batsman_df[batsman_df['batsman_id'].astype(str).isin(team1_ids)][batsman_stats_cols].mean()
    team2_batsman_avg = batsman_df[batsman_df['batsman_id'].astype(str).isin(team2_ids)][batsman_stats_cols].mean()
    team1_bowler_avg = bowler_df[bowler_df['bowler_id'].astype(str).isin(team1_ids)][bowler_stats_cols].mean()
    team2_bowler_avg = bowler_df[bowler_df['bowler_id'].astype(str).isin(team2_ids)][bowler_stats_cols].mean()

    # Convert averages to numpy arrays
    team1_batsman_avg = team1_batsman_avg.to_numpy()
    team2_batsman_avg = team2_batsman_avg.to_numpy()
    team1_bowler_avg = team1_bowler_avg.to_numpy()
    team2_bowler_avg = team2_bowler_avg.to_numpy()

    # Calculate ratios with safe division using numpy
    for i, stat in enumerate(batsman_stats_cols):
        ratio = np.divide(team1_batsman_avg[i], team2_batsman_avg[i], out=np.zeros_like(team1_batsman_avg[i]), where=team2_batsman_avg[i] != 0)
        match_df.at[index, f'ratio_{stat}'] = ratio
        print(f"ratio_{stat}: {match_df.at[index, f'ratio_{stat}']}")

    for i, stat in enumerate(bowler_stats_cols):
        ratio = np.divide(team1_bowler_avg[i], team2_bowler_avg[i], out=np.zeros_like(team1_bowler_avg[i]), where=team2_bowler_avg[i] != 0)
        match_df.at[index, f'ratio_{stat}'] = ratio
        print(f"ratio_{stat}: {match_df.at[index, f'ratio_{stat}']}")

# Handle missing values by replacing them with 0 (or any other desired value)
match_df.fillna(0, inplace=True)

# Save the final dataset
final_csv_path = 'final_train_data_with_features.csv'
match_df.to_csv(final_csv_path, index=False)
print(f'Final dataset saved to {final_csv_path}')