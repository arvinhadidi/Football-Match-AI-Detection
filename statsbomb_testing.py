import pandas as pd
from statsbombpy import sb
import os

# Replace with a valid match_id from the free open data
matchId = 19809

# Fetch the events DataFrame for the match
events_df = sb.events(match_id = matchId)

# Filter for only shot events (the event type is in the 'type_name' column)
shots_df = events_df[events_df['type'] == 'Shot'].copy()

desired_columns = [
    'shot_statsbomb_xg',   # Expected Goals
    'play_pattern',   # Play pattern
    'position',       # Player's position
    'location',            # [x, y] coordinates of the shot
    'under_pressure',      # Whether the shot was taken under pressure
    'shot_aerial_won',     # Outcome of aerial duel (if applicable)
    #'follows_dribble',     # Whether the shot followed a dribble (if available)
    'shot_first_time',     # Whether the shot was taken first time
    'open_goal',           # Whether it was an open goal
    'deflected',           # Whether the shot was deflected
    'shot_technique',      # Technique used for the shot
    'body_part_name',      # Body part used (e.g., left foot, head)
    'shot_one_on_one',
    'goalkeeper_position',
    'goalkeeper_technique',
]

# Not all columns may exist; filter for the ones present
available_columns = [col for col in desired_columns if col in shots_df.columns]

# Create a new DataFrame with only the desired columns
shots_filtered = shots_df[available_columns]

# Save to a temporary CSV file
filename = "temp_xg_data.csv"
shots_filtered.to_csv(filename, index=False)

# Open the file automatically
os.system(f"open {filename}")  # Works on macOS

# Display the first few rows of the filtered shots DataFrame
# print(shots_filtered)
