import pandas as pd

feat_df = pd.DataFrame(columns=['feat_id', 'feat_name', 'model_feat_imp_train', 'feat_rank_train', 'feat_description'])
feat_df['feat_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
feat_df['feat_name'] = ['team1_id', 'team2_id', 'toss winner', 'toss decision', 'lighting', 'series_name', 'season', 'ground_id', 'team_count_50runs_last15', 'team_winp_last5']
feat_df['feat_rank_train'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
feat_df['feat_description'] = ['ID of team 1', 
                               'ID of team 2', 
                               'Which team won the toss', 
                               'What was the decision of the toss winner, bat or field?', 
                               'What were the ground lighting conditions',
                               'What was the name of the series',
                               'What was the game season',
                               'ID of ground',
                               'Ratio of number of 50s by players in team1 to number of 50s by players in team2 in last 15 games',
                               'Ratio of team1 win % to team2 win % in last 5 games']
file_path = 'feature.csv'
feat_df.to_csv(file_path, index=False)