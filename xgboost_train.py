import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import csv

#prepare train data
original_train_data = pd.read_csv('train.csv')
columns_to_drop = ['match id', 'match_dt', 'team1', 'team2', 'winner', 'venue', 'city', 'team1_roster_ids', 'team2_roster_ids']
train_data = original_train_data.drop(columns=columns_to_drop)
categorical_columns = ['toss winner', 'toss decision', 'lighting', 'series_name', 'ground_id', 'season'] 
train_data[categorical_columns] = train_data[categorical_columns].astype('category')
team1_id = train_data['team1_id']
team2_id = train_data['team2_id']
train_data['winner_id'] = train_data.apply(lambda row: 0 if row['winner_id'] == row['team1_id'] else 1, axis=1) 
train_y = train_data['winner_id'] 
train_X = train_data.drop(columns=['winner_id'])

#prepare test data
original_test_data = pd.read_csv('test.csv')
columns_to_drop2 = ['match id', 'match_dt', 'team1', 'team2', 'venue', 'city', 'team1_roster_ids', 'team2_roster_ids']
test_data = original_test_data.drop(columns=columns_to_drop2)
test_data[categorical_columns] = test_data[categorical_columns].astype('category')
test_X = test_data

#XGBoost model
dtrain = xgb.DMatrix(train_X, label = train_y, enable_categorical = True)
dtest = xgb.DMatrix(test_X, enable_categorical = True)
param = {'booster': 'dart',
         'max_depth': 5, 
         'learning_rate': 0.135,
         'objective': 'binary:logistic',
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.24,
         'skip_drop': 0.84,
         'colsample_bytree': 0.805,
         'gamma': 0.225,
         'min_child_weight': 2.132,
         'n_estimators': 250,
         'subsample': 0.925}
num_round = 500
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)
featuer_imp = bst.get_score(importance_type='gain')
traind = xgb.DMatrix(train_X, enable_categorical = True)
preds_train = bst.predict(traind)

#printing submission files
#creating pandas dataframe for feature file
feat_df = pd.DataFrame(columns=['feat-id', 'feat_name', 'model_feat_imp_train', 'feat_rank_train', 'feat_description'])
#creating pandas dataframe for result file
result_df = pd.DataFrame(columns=["match id", "dataset_type", "win_pred_team_id", "win_pred_score", "indep_feat_id1", "indep_feat_id2", "indep_feat_id3", "indep_feat_id4", "indep_feat_id5", "indep_feat_id6", "indep_feat_id7", "indep_feat_id8", "indep_feat_id9", "indep_feat_id10", "train_algorithm", "is_ensemble", "train_hps_trees", "train_hps_depth", "train_hps_lr"])
matches = list(original_test_data['match id']).append(list(original_train_data['match id']))
result_df['match id'] = matches
#printing the winning team and predicted chances
match_id = []
win_id_test = []
chances_test = []
for i in range(0, len(preds)):
    if preds[i]>0.5:
        chances_test.append(preds[i])
        win_id_test.append(original_test_data['team2_id'].iloc[i])
    else:
        chances_test.append(1-preds[i])
        win_id_test.append(original_test_data['team1_id'].iloc[i])
    match_id.append(original_test_data['match id'].iloc[i])
win_id_train = []
chances_train = []
for i in range(0, len(preds_train)):
    if preds_train[i]>0.5:
        chances_train.append(preds_train[i])
        win_id_train.append(original_train_data['team2_id'].iloc[i])
    else:
        chances_train.append(1-preds_train[i])
        win_id_train.append(original_train_data['team1_id'].iloc[i])
    match_id.append(original_train_data['match id'].iloc[i])
win_id = win_id_test + win_id_train
chances = chances_test + chances_train
result_df['win_pred_team_id'] = win_id
result_df['win_pred_score'] = chances
result_df['match id'] = match_id
#printing imp independent features
indep_feat = {k: v for k, v in sorted(featuer_imp.items(), key=lambda item: item[1])}
keys = list(featuer_imp.keys())
result_df['indep_feat_id1'] = keys[0]
result_df['indep_feat_id2'] = keys[1]
result_df['indep_feat_id3'] = keys[2]
result_df['indep_feat_id4'] = keys[3]
result_df['indep_feat_id5'] = keys[4]
result_df['indep_feat_id6'] = keys[5]
result_df['indep_feat_id7'] = keys[6]
result_df['indep_feat_id8'] = keys[7]
result_df['indep_feat_id9'] = keys[8]
result_df['indep_feat_id10'] = keys[9]
result_df['indep_feat_id11'] = keys[10]
result_df['indep_feat_id12'] = keys[11]
result_df['indep_feat_id13'] = keys[12]
result_df['indep_feat_id14'] = keys[13]
result_df['indep_feat_id15'] = keys[14]

#filling remaining columns
result_df.loc[:, 'dataset_type'] = 'r1'
result_df.loc[:, 'train_algorithm'] = 'XGBoost'
result_df.loc[:, 'is_ensemble'] = 'no'
result_df.loc[:, 'train_hps_trees'] = 250
result_df.loc[:, 'train_hps_depth'] = 5
result_df.loc[:, 'train_hps_lr'] = 0.135
print(result_df)
#convert pandas dataframe to csv file
file_path1 = 'result_xgb.csv'
file_path2 = 'feature.csv'
result_df.to_csv(file_path1, index=False)
#feat_df.to_csv(file_path2, index=False)