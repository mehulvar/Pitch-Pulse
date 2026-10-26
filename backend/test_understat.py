import pandas as pd
from understatapi import UnderstatClient

with UnderstatClient() as understat:
    # Every Premier League match of one season.
    # Understat names seasons by the year they start, so "2016" means 2016/17
    matches = understat.league(league="EPL").get_match_data(season="2016")
    print("Matches found:", len(matches))
    # Transposed so each column name is easy to read down the page
    print(pd.DataFrame(matches).head(2).T)

    # Shots for one match. Each match has its own id in the list above
    match_id = matches[0]["id"]
    shots = understat.match(match=match_id).get_shot_data()

    # Understat splits shots into home ("h") and away ("a")
    home_shots = pd.DataFrame(shots["h"])
    print("Home shots in that match:", len(home_shots))
    print(home_shots.head(2).T)