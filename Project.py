import csv
import math
import re
import pandas as pd
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit,leastsq
from collections import defaultdict
from matplotlib import pyplot
from scipy.linalg import svd


donor_pool_size = 750
T0=30
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

def generate_donor_pool_and_treatment_units(matches_map, donor_pool_size):
    donor_pool = []
    treatment_units =[]
    count=0
    for match_id in matches_map:
        count+=1
        if count<=donor_pool_size:
            for innings in range(1,3,1):
                donor_pool_runs_row=[]
                donor_pool_wickets_row=[]
                for wickets_left in range(10,-1,-1):
                    overs,runs = filter_data(wickets_left,matches_map,str(innings),match_id)
                    for i in range(len(overs)):
                        donor_pool_runs_row.append(runs[i])
                        donor_pool_wickets_row.append(10-wickets_left)

                n = len(donor_pool_runs_row)
                for i in range(n,50,1):
                    donor_pool_runs_row.append(donor_pool_runs_row[n-1])
                    donor_pool_wickets_row.append(donor_pool_wickets_row[n-1])
                donor_pool.append(donor_pool_runs_row+donor_pool_wickets_row)
        else:
            for innings in range(1,3,1):
                treatment_units_runs_row=[]
                treatment_units_wickets_row=[]
                for wickets_left in range(10,-1,-1):
                    overs,runs = filter_data(wickets_left,matches_map,str(innings),match_id)

                    for i in range(len(overs)):
                        if (len(treatment_units_runs_row) >=T0):
                            break
                        treatment_units_runs_row.append(runs[i])
                        treatment_units_wickets_row.append(10-wickets_left)

                n = len(treatment_units_runs_row)
                for i in range(n,T0,1):
                    treatment_units_runs_row.append(treatment_units_runs_row[n-1])
                    treatment_units_wickets_row.append(treatment_units_wickets_row[n-1])
                treatment_units.append(treatment_units_runs_row+treatment_units_wickets_row)


    return donor_pool,treatment_units

def denoise_donor_pool(donor_pool):
    U,S,VT = svd(donor_pool)
    print(len(U),len(S),len(VT))
    #TODO
    return donor_pool

def objective(x,weights):
    predicted_runs=0
    for i in range(len(x)):
        predicted_runs+=x[i]*weights
    return predicted_runs

def transpose(matrix):
    return tuple(zip(*matrix))

def predict_scores(denoised_donor_pool, treatment_unit):

    ###INITIAL VERSION WITH JUST ONE SINGLE WEIGHT PARAMETER

    runs_label=treatment_unit[0:30]
    wickets_label=treatment_unit[30:60]
    inputs_runs=[]
    input_wickets=[]

    test_input_runs=[]
    test_input_wickets=[]
    for i in range(T0):
        feature_vector_runs=[]
        feature_vector_wickets=[]
        predict_runs_feature_vector=[]
        predict_wickets_feature_vector=[]
        for j in range(len(denoised_donor_pool)):
            feature_vector_runs.append(donor_pool[j][i])
            feature_vector_wickets.append(donor_pool[j][i+50])
            if(i>=10):
                predict_runs_feature_vector.append(donor_pool[j][i+20])
                predict_wickets_feature_vector.append(donor_pool[j][i+70])

        inputs_runs.append(feature_vector_runs)
        input_wickets.append(feature_vector_wickets)
        if(i>=10):
            test_input_runs.append(predict_runs_feature_vector)
            test_input_wickets.append(predict_wickets_feature_vector)

    popt1,_ = curve_fit(objective, transpose(inputs_runs), runs_label,maxfev=1000000)
    popt2,_ = curve_fit(objective, transpose(input_wickets), wickets_label,maxfev=1000000)

    final_treatment_unit_runs=treatment_unit[0:30]
    final_treatment_unit_wickets=treatment_unit[30:60]
    for i in range(len(test_input_runs)):
        final_treatment_unit_runs.append(objective(test_input_runs[i],popt1)[0])
        final_treatment_unit_wickets.append(objective(test_input_wickets[i],popt2)[0])

    print(final_treatment_unit_runs)
    print(final_treatment_unit_wickets)

    #TODO
    return 0

if __name__ == '__main__':
    matches_map = clean_data(input_data)
    #Step1: Concatenation
    print('-----GENERATING DONOR POOL AND TREATMENT UNITS-----')
    donor_pool,treatment_units = generate_donor_pool_and_treatment_units(matches_map, donor_pool_size)
    print('DONOR POOL SIZE: ',len(donor_pool))
    print('TREATMENT UNITS SIZE: ',len(treatment_units))
    print('LENGTH OF TREATMENT UNITS: ',len(treatment_units[0]))
    print('-----DONOR POOL AND TREATMENT UNITS GENERATED!!-----')

    #Step2: Denoising
    print('-----DENOISING DONOR POOL-----')
    denoised_donor_pool = denoise_donor_pool(donor_pool)
    print('-----DENOISING DONE!!-----')

    #Step3: Linear Regression AND Prediction
    print('-----STARTING LINEAR REGRESSION AND PREDICTION-----')
    for treatment_unit in treatment_units:
        predicted_scores = predict_scores(denoised_donor_pool, treatment_unit)
        ####UNCOMMENT
        break
    print('-----LINEAR REGRESSION AND PREDICTION DONE!!-----')









