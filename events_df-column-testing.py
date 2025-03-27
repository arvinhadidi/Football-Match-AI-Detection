from statsbombpy import sb

# Replace with a valid match_id from the free open data
matchId = 15946  

# Fetch the events DataFrame for the match
events_df = sb.events(match_id = matchId)

print(events_df.columns)