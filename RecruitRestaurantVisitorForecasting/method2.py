
'''
This method uses these features
['dow', 'year', 'month', 'day_of_week', 'holiday_flg', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors', 'count_observations', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'rs1_x', 'rv1_x', 'rs2_x', 'rv2_x', 'rs1_y', 'rv1_y', 'rs2_y', 'rv2_y', 'total_reserv_sum', 'total_reserv_mean', 'total_reserv_dt_diff_mean']
RMSE GradientBoostingRegressor:  0.501477019571
RMSE KNeighborsRegressor:  0.421517079307
'''
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

data = {
    'tra': pd.read_csv('./data/air_visit_data.csv'),
    'as': pd.read_csv('./data/air_store_info.csv'),
    'hs': pd.read_csv('./data/hpg_store_info.csv'),
    'ar': pd.read_csv('./data/air_reserve.csv'),
    'hr': pd.read_csv('./data/hpg_reserve.csv'),
    'id': pd.read_csv('./data/store_id_relation.csv'),
    'tes': pd.read_csv('./data/sample_submission.csv'),
    'hol': pd.read_csv('./data/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }
# add 'air_store_id' to the last of data['hr']
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    # get year, month, day, get rid of time
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])


data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date


unique_stores = data['tes']['air_store_id'].unique()
# count week
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'week_min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'week_mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'week_median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'week_max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'week_count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

# count all
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].min().rename(columns={'visitors':'all_min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].mean().rename(columns={'visitors':'all_mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].median().rename(columns={'visitors':'all_median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].max().rename(columns={'visitors':'all_max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].count().rename(columns={'visitors':'all_count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])

# count year
stores1 = pd.concat([pd.DataFrame({'air_store_id': unique_stores})], axis=0, ignore_index=True).reset_index(drop=True)
data2016 = data['tra'][data['tra']['year'].isin([2016])]
data2017 = data['tra'][data['tra']['year'].isin([2017])]
# count 2016
tmp = data2016.groupby(['air_store_id','year'], as_index=False)['visitors'].min().rename(columns={'visitors':'2016_min_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2016.groupby(['air_store_id','year'], as_index=False)['visitors'].mean().rename(columns={'visitors':'2016_mean_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2016.groupby(['air_store_id','year'], as_index=False)['visitors'].median().rename(columns={'visitors':'2016_median_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2016.groupby(['air_store_id','year'], as_index=False)['visitors'].max().rename(columns={'visitors':'2016_max_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2016.groupby(['air_store_id','year'], as_index=False)['visitors'].count().rename(columns={'visitors':'2016_count_observations'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
# count 2017
tmp = data2017.groupby(['air_store_id','year'], as_index=False)['visitors'].min().rename(columns={'visitors':'2017_min_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2017.groupby(['air_store_id','year'], as_index=False)['visitors'].mean().rename(columns={'visitors':'2017_mean_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2017.groupby(['air_store_id','year'], as_index=False)['visitors'].median().rename(columns={'visitors':'2017_median_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2017.groupby(['air_store_id','year'], as_index=False)['visitors'].max().rename(columns={'visitors':'2017_max_visitors'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])
tmp = data2017.groupby(['air_store_id','year'], as_index=False)['visitors'].count().rename(columns={'visitors':'2017_count_observations'})
stores1 = pd.merge(stores1, tmp, how='left', on=['air_store_id'])

stores = pd.merge(stores, stores1, how='left', on=['air_store_id'])
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(train, stores, how='left', on=['air_store_id','dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



for df in ['ar','hr']:
    # data[df].to_csv(df + '.csv')
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)
train.to_csv('train.csv')
test.to_csv('test.csv')

print('start to train...')
model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2)
model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
model1.fit(train[col], np.log1p(train['visitors'].values))
model2.fit(train[col], np.log1p(train['visitors'].values))
print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), model1.predict(train[col])))
print('RMSE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), model2.predict(train[col])))
test['visitors'] = (model1.predict(test[col]) + model2.predict(test[col])) / 2
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id','visitors']].copy()
sub1[['id', 'visitors']].to_csv('submit2.csv', index=False)












