import csv
import math
import re
import pandas as pd
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit,leastsq
from collections import defaultdict
from matplotlib import pyplot


donor_pool_size = 750
input_data = 'data/04_cricket_1999to2011.csv'
##Filtering columns of interest manually
use_cols = ['Match','Innings','Over','Runs','Total.Runs','Innings.Total.Runs','Runs.Remaining','Total.Out','Outs.Remaining','Wickets.in.Hand']
matches_map=defaultdict(list)

def read_csv(filename):
    return pd.read_csv(filename, sep=',',usecols=use_cols)

def transpose(matrix):
    return tuple(zip(*matrix))

##Initial preprocessing of data
def pre_process(data):
    transformed_data=[]
    transformed_data.append(data['Match'])
    transformed_data.append(data['Innings'])
    transformed_data.append(data['Over'])
    transformed_data.append(data['Runs'])
    transformed_data.append(data['Total.Runs'])
    transformed_data.append(data['Innings.Total.Runs'])
    transformed_data.append(data['Runs.Remaining'])
    transformed_data.append(data['Total.Out'])
    transformed_data.append(data['Outs.Remaining'])
    transformed_data.append(data['Wickets.in.Hand'])
    return transformed_data

##Generating matchId -> [match_data] map
def generate_matches_map(data):
    main_map=defaultdict(list)
    for column in range(0,len(data[0])):
        temp_data=[]
        for row in range (1,10):
            temp_data.append(str(data[row][column]))
        main_map[str(data[0][column])].append(temp_data)
    return main_map

def filter_data(wicket_left, data,inning,match_id):
    x=[]
    y=[]
    for entry in data[match_id]:
        if (inning==str(entry[0]) and wicket_left==int(entry[8])):
            overs=int(entry[1])
            runs=int(entry[3])
            if runs<0:
                runs=0
            x.append(overs)
            y.append(runs)
    return x,y

def clean_data(input_data):
    global matches_map
    data=read_csv(input_data)
    transformed_data=pre_process(data)
    matches_map=generate_matches_map(transformed_data)
    return matches_map

def generate_donor_pool(matches_map,donor_pool_size):
    donor_pool = []
    count=0
    for match_id in matches_map:
        if(count==donor_pool_size):
            break
        count+=1
        for innings in range(1,3,1):
            donor_pool_runs_row=[]
            donor_pool_wickets_row=[]
            for wickets_left in range(10,-1,-1):
                overs,runs = filter_data(wickets_left,matches_map,str(innings),match_id)
                for i in range(len(overs)):
                    donor_pool_runs_row.append(runs[i])
                    donor_pool_wickets_row.append(10-wickets_left)

            for i in range(len(donor_pool_runs_row),50,1):
                donor_pool_runs_row.append(-1)
                donor_pool_wickets_row.append(-1)
            donor_pool.append(donor_pool_runs_row+donor_pool_wickets_row)

    return donor_pool

def denoise_donor_pool(donor_pool):
    #TODO
    return 0

def linear_regression(denoised_donor_pool):
    #TODO
    return 0

if __name__ == '__main__':
    matches_map = clean_data(input_data)
    #Step1: Concatenation
    print('-----GENERATING DONOR POOL-----')
    donor_pool = generate_donor_pool(matches_map,donor_pool_size)
    print('DONOR POOL SIZE: ',len(donor_pool))
    print('-----DONOR POOL GENERATED!!-----')

    #Step2: Denoising
    print('-----DENOISING DONOR POOL-----')
    denoised_donor_pool = denoise_donor_pool(donor_pool)
    print('-----DENOISING DONE!!-----')

    #Step3: Linear Regression
    print('-----STARTING LINEAR REGRESSION-----')
    weights = linear_regression(denoised_donor_pool)
    print('-----LINEAR REGRESSION DONE!!-----')

    #RUN PREDICTOR








