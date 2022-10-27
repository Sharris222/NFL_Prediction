# NFL_Prediction
- Variation of the NFL predictor created in the following article: https://www.activestate.com/blog/how-to-predict-nfl-winners-with-python/
- Repo of original script: https://gitlab.com/dsblendo/nfl_predicting_game_outcomes

Following information needs to be updated weekly:
- nfl_elo.csv found in the get_elo() function. Can download this at https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv
- the line "elo_df = elo_df[elo_df.date < '11-03-2022']" in the get_elo() function. The date needs to be set to the Thursday of the following week. Example, when running the script for week 8 of the 2022 season, the date should be set to '11-03-2022' because the Thursday games of week 9 are on '11-03-2022'.
- the current_week in the main() function needs to be set to the week of the 2022 season that you want to make predictions for.
- the line "test_df = agg_games_df.tail(15)" needs to be updated with the number of games for the current week.

Note:
- Changing the year in the first line of main() will directly impact the amount of training data (year always needs to be <= 2022). The further away the year is from 2022, the longer the script takes.
- Script uses linear regression to make predictions, can tune the linear regression parameters as you see fit.

Week 7 results:
Accurately picked every winner (team in each game with prediction of 51% chance or more to win) except Panthers, Commanders, and Bears.

How to run script: `python predict.py`

