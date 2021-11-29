import csv
import math
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import arange
from scipy.optimize import curve_fit,least_squares
from collections import defaultdict
from scipy.linalg import svd

ERROR_RUNS=0
ERROR_WKTS=0
n_iter=10
st=30
en=31
init_guess=0.00001
donor_pool_size = 750
L=1000
model_scores=[]
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

def generate_donor_pool_and_treatment_units(matches_map, donor_pool_sizex):
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
    donor_pool=np.array(donor_pool)
    U,S,VT = svd(donor_pool)
    print("DonorPoolShaPE----",donor_pool.shape)
    #THRESHOLDING BASED ON SINGULAR VALUES>1000
    sprime=S[S>L]
    zeroes=np.zeros(len(S)-len(sprime))
    sprime=np.append(sprime,zeroes)
    sprime_mat=np.diag(sprime)
    #SIGMA SHOULD BE 1500X100 HENCE NEED TO ADD 0S
    zeros2=np.zeros((2*donor_pool_size-100,100))
    s_mat_new=np.append(sprime_mat,zeros2,axis=0)
    temp=np.dot(U,s_mat_new)
    denoised_donor_pool=np.dot(temp,VT)
    print("DenoisedDonorPoolShaPE----",denoised_donor_pool.shape)


    return denoised_donor_pool


def denoise_donor_pool_pca(donor_pool):
    donor_pool=np.array(donor_pool)
    U,S,VT = svd(transpose(donor_pool) * donor_pool)
    print("DonorPoolShaPE----",donor_pool.shape)
    x,y=donor_pool.shape
    #THRESHOLDING BASED ON SINGULAR VALUES>1000
    sprime=S[S>L]
    zeroes_2 = np.zeros((y,y))
    for i in range (y):
        zeroes_2[i,i]= sprime[i]
    temp=np.dot(U,zeroes_2)
    denoised_donor_pool=np.dot(temp,VT)
    print("DenoisedDonorPoolShaPE----",denoised_donor_pool.shape)
    return denoised_donor_pool

def objective(weights,x):
    predicted_runs=0
    for i in range(len(x)):
        predicted_runs+=x[i]*weights[i]
    return predicted_runs

def error_func(weights,x,y):
    return (objective(weights,x)-y)**2

def predict_scores(denoised_donor_pool, treatment_unit,t0,target):
    actual_innings=[]
    predicted_innings=[]
    last_row=[]
    for i in range (100):
        if(i<50):
            last_row.append(target)
        else:
            last_row.append(10)
    denoised_donor_pool=np.vstack([denoised_donor_pool,last_row])

    runs_label=treatment_unit[0:t0]
    wickets_label=treatment_unit[50:50+t0]
    inputs_runs=[]
    input_wickets=[]

    test_input_runs=[]
    test_input_wickets=[]
    for i in range(t0,50):
        predict_runs_feature_vector=[]
        predict_wickets_feature_vector=[]
        for j in range(len(denoised_donor_pool)):
            predict_runs_feature_vector.append(denoised_donor_pool[j][i])
            predict_wickets_feature_vector.append(denoised_donor_pool[j][i+50])

        test_input_runs.append(predict_runs_feature_vector)
        test_input_wickets.append(predict_wickets_feature_vector)


    for i in range(t0):
        feature_vector_runs=[]
        feature_vector_wickets=[]
        for j in range(len(denoised_donor_pool)):
            feature_vector_runs.append(denoised_donor_pool[j][i])
            feature_vector_wickets.append(denoised_donor_pool[j][i+50])

        inputs_runs.append(feature_vector_runs)
        input_wickets.append(feature_vector_wickets)

    final_weights_runs = fit_data(inputs_runs, runs_label)

    final_weights_wickets = fit_data(input_wickets, wickets_label)

    final_treatment_unit_runs=[]
    final_treatment_unit_wickets=[]

    for i in range(t0):
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

def plot(actual_innings,predicted_innings):
    x=[]
    for i in range(50):
        x.append(i+1)
    plt.xlabel('Overs')
    plt.title('mRSC Match Prediction without denoising')
    plt.ylabel('Runs Scored')
    plt.xticks(rotation=90)
    plt.plot(x,actual_innings[0],label='Actual Match Runs')
    plt.plot(x,predicted_innings[0],label='Model Predicted Match Runs')
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.show()

    plt.xlabel('Overs')
    plt.title('mRSC Match Prediction without denoising')
    plt.ylabel('Wkts Lost')
    plt.xticks(rotation=90)
    plt.scatter(x,actual_innings[1],label='Actual Match Wkts(Cumulative)',s=40)
    plt.scatter(x,predicted_innings[1],label='Model Predicted Match Wkts(Cumulative)',s=10)
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.show()


def update_score(vector):
    print(vector)
    model_scores.append(vector)


def post_process_prediction(actual_innings,predicted_innings):
    #plot(actual_innings,predicted_innings)
    calculate_error_and_probability(actual_innings, predicted_innings)



def calculate_error_and_probability(actual_innings, predicted_innings):
    global ERROR_RUNS
    global ERROR_WKTS
    actual_runs_list = np.array(actual_innings[0])
    predicted_runs_list = np.array(predicted_innings[0])
    actual_wickets_list = np.array(actual_innings[1])
    predicted_wickets_list = np.array(predicted_innings[1])
    runs_error_squared_list = (actual_runs_list - predicted_runs_list) ** 2
    wickets_error_squared_list = (actual_wickets_list - predicted_wickets_list) ** 2
    print('TOTAL RUNS ERROR:')
    print(np.average(runs_error_squared_list))
    ERROR_RUNS+=np.average(runs_error_squared_list)
    print('TOTAL WKTS ERROR:')
    print(np.average(wickets_error_squared_list))
    ERROR_WKTS+=np.average(wickets_error_squared_list)

def get_predicted_result(predicted_innings, target):
    #returns 1 if its a win
    return get_result(predicted_innings, target)

def get_actual_result(actual_innings,target):
    #returns 1 if its a win
    return get_result(actual_innings, target)

def get_result(innings, target):
    for i in range(len(innings[0])):
        if (innings[1][i] >= 10):
            if (innings[0][i] >= target):
                return 1
            return 0
        if (innings[0][i] >= target):
            return 1
    return 0


def dump_op():
    print('ACTUAL RUNS:')
    print(actual_innings[0])
    print('PREDICTED RUNS:')
    print(predicted_innings[0])
    print('ACTUAL WKTS:')
    print(actual_innings[1])
    print('PREDICTED WKTS:')
    print(predicted_innings[1])

def plot_model_scores():
    x=[]
    y=[]
    for i in range(st,en):
        x.append(i)
    for j in range(len(model_scores[0])):
        total=0
        for i in range(len(model_scores)):
            total+=model_scores[i][j]
        y.append(total)

    plt.xlabel('Overs')
    plt.title('mRSC Performance Plot')
    plt.ylabel('Model Scores')
    plt.xticks(rotation=90)
    plt.plot(x,y,label='Model Scores')
    plt.tight_layout()

    plt.show()


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
    N=0
    for i in range(N,N+2):
        if(i%2==0):
            continue
        target = treatment_units[i-1][49]+1
        scores=[]
        for overs in range(st,en):
            print(overs)
            actual_innings,predicted_innings=predict_scores(denoised_donor_pool, treatment_units[i],overs,target)
            dump_op()
            post_process_prediction(actual_innings,predicted_innings)
            actual_result = get_actual_result(actual_innings,target)
            predicted_result= get_predicted_result(predicted_innings, target)
            correct=0
            print(actual_result,predicted_result)
            if actual_result==predicted_result:
                correct=1
            scores.append(correct)
        update_score(scores)
    plot_model_scores()

print('-----LINEAR REGRESSION AND PREDICTION DONE!!-----')

print('FINAL ERROR IN RUNS: ', ERROR_RUNS)
print('FINAL ERROR IN WKTS: ', ERROR_WKTS)

'''
from sklearn.decompositionimport PCA pca= PCA(n_components= 2) smooth_donor= pca.fit_transform(Donor)
'''
