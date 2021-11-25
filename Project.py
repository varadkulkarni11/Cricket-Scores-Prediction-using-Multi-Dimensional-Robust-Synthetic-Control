import csv
import math
import re
import pandas as pd
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit,least_squares
from collections import defaultdict
from matplotlib import pyplot
from scipy.linalg import svd

n_iter=10
init_guess=0.00001
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
             generate_donor_pool(donor_pool, match_id, matches_map)
        else:
            generate_treatment_units(match_id, matches_map, treatment_units)


    return donor_pool,treatment_units


def generate_treatment_units(match_id, matches_map, treatment_units):
    for innings in range(1, 3, 1):
        treatment_units_runs_row = []
        treatment_units_wickets_row = []
        for wickets_left in range(10, -1, -1):
            overs, runs = filter_data(wickets_left, matches_map, str(innings), match_id)

            for i in range(len(overs)):
                treatment_units_runs_row.append(runs[i])
                treatment_units_wickets_row.append(10 - wickets_left)

        n = len(treatment_units_runs_row)
        for i in range(n, 50, 1):
            treatment_units_runs_row.append(treatment_units_runs_row[n - 1])
            treatment_units_wickets_row.append(treatment_units_wickets_row[n - 1])
        treatment_units.append(treatment_units_runs_row + treatment_units_wickets_row)


def generate_donor_pool(donor_pool, match_id, matches_map):
    for innings in range(1, 3, 1):
        donor_pool_runs_row = []
        donor_pool_wickets_row = []
        for wickets_left in range(10, -1, -1):
            overs, runs = filter_data(wickets_left, matches_map, str(innings), match_id)
            for i in range(len(overs)):
                donor_pool_runs_row.append(runs[i])
                donor_pool_wickets_row.append(10 - wickets_left)

        n = len(donor_pool_runs_row)
        for i in range(n, 50, 1):
            donor_pool_runs_row.append(donor_pool_runs_row[n - 1])
            donor_pool_wickets_row.append(donor_pool_wickets_row[n - 1])
        donor_pool.append(donor_pool_runs_row + donor_pool_wickets_row)


def denoise_donor_pool(donor_pool):
    U,S,VT = svd(donor_pool)
    print(len(U),len(S),len(VT))
    #TODO
    return donor_pool

def objective(weights,x):
    predicted_runs=0
    for i in range(len(x)):
        predicted_runs+=x[i]*weights[i]
    return predicted_runs

def error_func(weights,x,y):
    return (objective(weights,x)-y)**2

def predict_scores(denoised_donor_pool, treatment_unit):
    actual_innings=[]
    predicted_innings=[]

    runs_label=treatment_unit[0:T0]
    wickets_label=treatment_unit[50:50+T0]
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

    final_weights_runs = fit_data(inputs_runs, runs_label)
    print(final_weights_runs[0:10])

    final_weights_wickets = fit_data(input_wickets, wickets_label)
    print(final_weights_wickets[0:10])

    final_treatment_unit_runs=[]
    final_treatment_unit_wickets=[]
    for i in range(30):
        final_treatment_unit_runs.append(math.floor(objective(inputs_runs[i],final_weights_runs)))
        final_treatment_unit_wickets.append(math.floor(objective(input_wickets[i],final_weights_wickets)))
    for i in range(len(test_input_runs)):
        final_treatment_unit_runs.append(math.floor(objective(test_input_runs[i],final_weights_runs)))
        final_treatment_unit_wickets.append(math.floor(objective(test_input_wickets[i],final_weights_wickets)))

    actual_innings.append(treatment_unit[0:50])
    actual_innings.append(treatment_unit[50:100])
    predicted_innings.append(final_treatment_unit_runs)
    predicted_innings.append(final_treatment_unit_wickets)

    return actual_innings,predicted_innings


def print_outputs(final_treatment_unit_runs, final_treatment_unit_wickets, treatment_unit):
    print('PREDICTED RUNS: ')
    print(final_treatment_unit_runs[30:50])
    print('ACTUAL RUNS:')
    print(treatment_unit[30:50])
    print('PREDICTED Wickets: ')
    print(final_treatment_unit_wickets[30:50])
    print('ACTUAL WICKETS:')
    print(treatment_unit[80:100])


def fit_data(input_vectors, labels):
    x = np.array(transpose(input_vectors))
    y = np.array(labels)
    p0 = [init_guess] * len(x)
    output = least_squares(error_func, x0=p0, bounds=(0, 1), args=(x, y), max_nfev=n_iter)
    final_weights = output.x
    return final_weights

def post_process_prediction(actual_innings,predicted_innings):
    return 0

if __name__ == '__main__':
    matches_map = clean_data(input_data)
    #Step1: Concatenation
    print('-----GENERATING DONOR POOL AND TREATMENT UNITS-----')
    donor_pool,treatment_units = generate_donor_pool_and_treatment_units(matches_map, donor_pool_size)
    print('DONOR POOL SIZE: ',len(donor_pool))
    print('TREATMENT UNITS SIZE: ',len(treatment_units))
    print('-----DONOR POOL AND TREATMENT UNITS GENERATED!!-----')

    #Step2: Denoising
    print('-----DENOISING DONOR POOL-----')
    denoised_donor_pool = denoise_donor_pool(donor_pool)
    print('-----DENOISING DONE!!-----')

    #Step3: Linear Regression AND Prediction
    print('-----STARTING LINEAR REGRESSION AND PREDICTION-----')
    N=len(treatment_units)
    N=2
    for i  in range(N):
        actual_innings,predicted_innings=predict_scores(denoised_donor_pool, treatment_units[i])
        print('ACTUAL RUNS:')
        print(actual_innings[0])
        print('PREDICTED RUNS:')
        print(predicted_innings[0])

        print('ACTUAL WKTS:')
        print(actual_innings[1])
        print('PREDICTED WKTS:')
        print(predicted_innings[1])

        post_process_prediction(actual_innings,predicted_innings)

    print('-----LINEAR REGRESSION AND PREDICTION DONE!!-----')









