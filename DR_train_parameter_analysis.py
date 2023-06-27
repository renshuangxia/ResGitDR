import copy
import csv
import shutil
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier

'''
    This script is to predict drug sensitivity using hidden representation from GIT-RINN which trained all data,
    this file also used all data as training data.
'''

'''
    print cpu information and set random seed
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(random_seed)


'''
    load data
'''
Drug_sensi_df = pd.read_csv("data/GDSC1_drug_response_binaryAUC.csv",index_col=0)
Drug_sensi_df.index = Drug_sensi_df.index.astype(str)
Drug_sensi_df_copy = Drug_sensi_df.copy()
print(Drug_sensi_df.shape)

gdsc_can_df = pd.read_csv("data/GDSC_cancer_type.csv",index_col=0)
gdsc_can_df.index = gdsc_can_df.index.astype(str)
gdsc_can_df = pd.get_dummies(gdsc_can_df) #change cancer type to one hot vector
print("gdsc_can_df",gdsc_can_df)


gdsc_mut_df = pd.read_csv("data/GDSC_sga_data.csv",index_col=0)
gdsc_mut_df.index = gdsc_mut_df.index.astype(str)
#common sample between drug sensitivity file and sga file
common_samples_list = [sample for sample in Drug_sensi_df.index if sample in gdsc_mut_df.index]
#select sga gene list
sga_gene_list = pd.read_csv("data/TCI_fondation1_driver_dataset_combine_gene_list.txt",header=None,names=["gene_name"])
gdsc_mut_df = gdsc_mut_df.loc[common_samples_list,sga_gene_list["gene_name"].values]
gdsc_mut_df.columns = ["sga_"+ gene for gene in gdsc_mut_df.columns]

'''
    load hyperparameter tuning results for Elastic Net
'''

hyperparameter_file_dir = "data/hyperparameters_tuning_result.csv"
hyperparameter_file = pd.read_csv(hyperparameter_file_dir,index_col=0)


data_dir= "result/parameters/ResGit/"


'''
    store result
'''
res_dir = "result/parameters/DR/"


#This function is to get all kinds of features for cell lines
def create_features(train_hidden_dir):
    train_features = {}
    train_features ["predicts_exprs"] = pd.read_csv(data_dir + "/results/predicts.csv", index_col=0)
    train_features ["predicts_exprs"].index = train_features ["predicts_exprs"].index.astype(str)
    GDSC_sample_index = []
    for sample_name in train_features ["predicts_exprs"].index:
        if "TCGA" not in str(sample_name):
            GDSC_sample_index.append(sample_name)
    curr_sample_name = [common_sample for common_sample in GDSC_sample_index if common_sample in common_samples_list]

    train_features ["sga"] = gdsc_mut_df.loc[curr_sample_name]
    train_features ["cancertype"] = gdsc_can_df.loc[curr_sample_name]
    train_features ["sga_cancertype"] = pd.concat([gdsc_can_df.loc[curr_sample_name],train_features["sga"]],axis=1)
    train_features ["predicts_exprs"] = train_features ["predicts_exprs"].loc[curr_sample_name]
    train_features ["scaled_targets_exprs"] = pd.read_csv(data_dir + "/results/targets.csv", index_col=0)
    train_features ["scaled_targets_exprs"].index = train_features ["scaled_targets_exprs"].index.astype(str)
    train_features ["scaled_targets_exprs"] = train_features ["scaled_targets_exprs"].loc[curr_sample_name]
    train_combined_features_df_list = []
    for idx in range(len(os.listdir(train_hidden_dir))):
        f = "hidden_outs_" + str(idx) + ".csv"
        prefix = "hidden_" + str(idx) + "_"
        curr_df = pd.read_csv(train_hidden_dir + "/" + f, index_col=0)
        curr_df.index = curr_df.index.astype(str)
        curr_df = curr_df.loc[curr_sample_name]
        cols = [prefix + str(i) for i in range(curr_df.shape[1])]
        curr_df.columns = cols
        train_features [f] = curr_df
        train_combined_features_df_list.append(curr_df)

    train_features ["all_hiddens"] = pd.concat(train_combined_features_df_list,axis=1)
    train_features ["sga_hiddens"] = pd.concat([train_features["all_hiddens"],train_features["sga"]],axis=1)

    return train_features

#make fold to store the result of drug sensitivity prediction
parameters_file = res_dir + "/parameters"

if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.makedirs(parameters_file)


#load hidden represention and predicted value from GIT-RINN
all_hidden_dir = data_dir  + "/hidden"
train_features = create_features(all_hidden_dir)
print(train_features ["sga_hiddens"])


# training the model
for i in range(0, Drug_sensi_df.shape[1]):
    col = Drug_sensi_df.iloc[:, i]  # current col
    drug = col.name
    print("Currnt drug ID: ", drug)

    nan_indicies = col.index[col.apply(np.isnan)]
    labeled_indicies = col.index[~col.apply(np.isnan)]  # remove rows with 'nan' values
    drug_labeled = col[labeled_indicies]

    res_dict = {"train" : {}, "test" : {}}
    for layer_file_name, hidden_rep_df in train_features.items():
        curr_train = pd.concat((hidden_rep_df, drug_labeled), axis=1, join='inner')
        if curr_train.shape[0] == 0:
            continue
        C = hyperparameter_file.iloc[i,1]
        l1_ratio = hyperparameter_file.iloc[i,2]
        # print(C,l1_ratio)
        model = LogisticRegression(penalty = 'elasticnet', solver = 'saga',C=C, l1_ratio = l1_ratio)
        model.fit(curr_train.iloc[:,:-1], curr_train.iloc[:, -1])

        if layer_file_name == "sga_hiddens":
            if i == 0:
                with open(os.path.join(parameters_file,layer_file_name),'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(["drug_id"] + hidden_rep_df.columns.to_list())
            with open(os.path.join(parameters_file,layer_file_name),'a') as f:
                writer = csv.writer(f)
                writer.writerow([drug] + list(model.coef_[0]))


