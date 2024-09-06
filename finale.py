import pandas as pd
import numpy as np

result_xgb = pd.read_csv('result_xgb.csv')
result_lgb = pd.read_csv('result_lgb.csv')
result_catboost = pd.read_csv('result_catboost.csv')

pred1 = result_xgb['win_pred_team_id'].to_numpy()
pred2 = result_lgb['win_pred_team_id'].to_numpy()
pred3 = result_catboost['win_pred_team_id'].to_numpy()

score1 = result_xgb['win_pred_score'].to_numpy()
score2 = result_lgb['win_pred_score'].to_numpy()
score3 = result_catboost['win_pred_score'].to_numpy()

final_pred = []
final_score = []

for i in range(0, len(pred1)):
    p1, p2, p3 = pred1[i], pred2[i], pred3[i]
    s1, s2, s3 = score1[i], score2[i], score3[i]
    if p1 == p2:
        final_pred.append(p1)
        f1 = (s1 + s2)/2
        final_score.append(f1)
    elif p1 == p3:
        final_pred.append(p1)
        f2 = (s1 + s3)/2
        final_score.append(f2)
    elif p2 == p3:
        final_pred.append(p2)
        f3 = (s2 + s3)/2
        final_score.append(f3)

       
result_df = pd.DataFrame(columns=[
    "match id", "dataset_type", "win_pred_team_id", "win_pred_score",
    "indep_feat_id1", "indep_feat_id2", "indep_feat_id3", "indep_feat_id4",
    "indep_feat_id5", "indep_feat_id6", "indep_feat_id7", "indep_feat_id8",
    "indep_feat_id9", "indep_feat_id10", "train_algorithm", "is_ensemble",
    "train_hps_trees", "train_hps_depth", "train_hps_lr"
])

result_df['match id'] = result_xgb['match id']
result_df['dataset_type'] = result_xgb['dataset_type']
result_df['win_pred_score'] = final_score
result_df['win_pred_team_id'] = final_pred
result_df['indep_feat_id1'] = result_xgb['indep_feat_id1']
result_df['indep_feat_id2'] = result_xgb['indep_feat_id2']
result_df['indep_feat_id3'] = result_xgb['indep_feat_id3']
result_df['indep_feat_id4'] = result_xgb['indep_feat_id4']
result_df['indep_feat_id5'] = result_xgb['indep_feat_id5']
result_df['indep_feat_id6'] = result_xgb['indep_feat_id6']
result_df['indep_feat_id7'] = result_xgb['indep_feat_id7']
result_df['indep_feat_id8'] = result_xgb['indep_feat_id8']
result_df['indep_feat_id9'] = result_xgb['indep_feat_id9']
result_df['indep_feat_id10'] = result_xgb['indep_feat_id10']
result_df['train_algorithm'] = 'XGBoost_LightGBM_CatBoost'
result_df['is_ensemble'] = 'yes'
result_df['train_hps_trees'] = result_xgb['train_hps_trees']
result_df['train_hps_depth'] = result_xgb['train_hps_depth']
result_df['train_hps_lr'] = result_xgb['train_hps_lr']

#convert pandas dataframe to csv file
file_path1 = 'result.csv'
result_df.to_csv(file_path1, index=False)