
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime

test = pd.read_csv('./submit2.1.csv')
sub1 = test[['id','visitors']].copy()

test = pd.read_csv('./submit2.2.csv')
sub2 = test[['id','visitors']].copy()
sub_merge = pd.merge(sub1, sub2, on='id', how='inner')

## Arithmetric Mean
sub_merge['visitors'] = (sub_merge['visitors_x'] + sub_merge['visitors_y']) / 2
sub_merge[['id', 'visitors']].to_csv('sub3_math_mean.csv', index=False)

## Geometric Mean
sub_merge['visitors'] = (sub_merge['visitors_x'] * sub_merge['visitors_y']) ** (1 / 2)
sub_merge[['id', 'visitors']].to_csv('sub3_geo_mean.csv', index=False)

## Harmonic Mean
sub_merge['visitors'] = 2 / (1 / sub_merge['visitors_x'] + 1 / sub_merge['visitors_y'])
sub_merge[['id', 'visitors']].to_csv('sub3_hrm_mean.csv', index=False)