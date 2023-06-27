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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

'''
    This script is to predict drug sensitivity using the hidden representation of ResGit, and it used
    cross-validation which training-testing dataset is same with GIT-RINN.
'''


'''
    print cpu information and set random seed
'''
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Use device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(random_seed)

'''
    set hyperparameters
'''

permute_labels = False


'''
    load data
'''
Drug_sensi_df = pd.read_csv("data/GDSC1_drug_response_binaryAUC.csv",index_col=0)
Drug_sensi_df.index = Drug_sensi_df.index.astype(str)
Drug_sensi_df_copy = Drug_sensi_df.copy()
print(Drug_sensi_df.shape)


if permute_labels == True:
    Drug_sensi_df = Drug_sensi_df.sample(frac=1).reset_index(drop=True)
    Drug_sensi_df = Drug_sensi_df.set_axis(Drug_sensi_df_copy.index)
    print(Drug_sensi_df)


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

'''
    load hidden representation data for each cell line
'''
data_dir= "result/cross_validation/ResGit/"

'''
    store result
'''
res_dir = "result/cross_validation/DR"

res_headers = ["drug", "sga_hiddens_f1","sga_hiddens_auroc"]

#This function is to get features for cell lines
def create_features(train_hidden_dir,phase):
    train_features = {}
    ResGit_preds = pd.read_csv(train_hidden_dir + "/" + phase+ "/results/predicts.csv", index_col=0)
    ResGit_preds.index = ResGit_preds.index.astype(str)
    GDSC_sample_index = []
    for sample_name in ResGit_preds.index.to_list():
        if "TCGA" not in str(sample_name):
            GDSC_sample_index.append(sample_name)
    curr_sample_name = list(set(GDSC_sample_index) &set(common_samples_list))

    train_features ["sga"] = gdsc_mut_df.loc[curr_sample_name]
    train_combined_features_df_list = []
    for idx in range(len(os.listdir(train_hidden_dir+ "/" + phase + "/hidden/"))):
        f = "hidden_outs_" + str(idx) + ".csv"
        prefix = "hidden_" + str(idx) + "_"
        curr_df = pd.read_csv(train_hidden_dir + "/" + phase + "/hidden/" + f, index_col=0)
        curr_df.index = curr_df.index.astype(str)
        curr_df = curr_df.loc[curr_sample_name]
        cols = [prefix + str(i) for i in range(curr_df.shape[1])]
        curr_df.columns = cols
        train_features [f] = curr_df
        train_combined_features_df_list.append(curr_df)
    train_features ["all_hiddens"] = pd.concat(train_combined_features_df_list,axis=1)
    train_features ["sga_hiddens"] = pd.concat([train_features["all_hiddens"],train_features["sga"]],axis=1)
    return train_features

feature = "sga_hiddens"

# cross validation
for fold_dir in os.listdir(data_dir):
    print("fold_dir",fold_dir)
    #make fold to store the result of drug sensitivity prediction
    train_res_file = res_dir + "/" + fold_dir + "/train.csv"
    test_res_file = res_dir + "/" + fold_dir + "/test.csv"
    # parameters_file = res_dir + "/" + fold_dir+ "/parameters"

    if os.path.exists(res_dir + "/" + fold_dir):
        shutil.rmtree(res_dir + "/" + fold_dir)

    os.makedirs(res_dir + "/" + fold_dir)
    with open(train_res_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(res_headers)
    with open(test_res_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(res_headers)

    #load hidden represention from ResGit
    train_hidden_dir = data_dir + fold_dir
    test_hidden_dir = data_dir + fold_dir

    train_features = create_features(train_hidden_dir,"train")
    test_features = create_features(test_hidden_dir,"test")
    print(test_features)

    '''
        Only select the data of the interested feature
    '''
    train_features = {key: train_features[key] for key in train_features.keys()}
    test_features = {key: test_features[key] for key in test_features.keys()}
    # print("curr_feature",test_features)


    train_preds = np.empty((train_features[feature].shape[0], Drug_sensi_df.shape[1]))
    train_preds[:] = np.nan

    test_preds = np.empty((test_features[feature].shape[0], Drug_sensi_df.shape[1]))
    test_preds[:] = np.nan

    train_preds = pd.DataFrame(train_preds, index=train_features[feature].index, columns=Drug_sensi_df.columns)

    train_targets = Drug_sensi_df.loc[train_preds.index, :]

    test_preds = pd.DataFrame(test_preds, index=test_features[feature].index, columns=Drug_sensi_df.columns)
    # print("test_preds",test_preds.index)
    test_targets = Drug_sensi_df.loc[test_preds.index, :]

    train_targets.to_csv(train_res_file.replace('train.csv', 'train_targets.csv'))
    test_targets.to_csv(test_res_file.replace('test.csv', 'test_targets.csv'))

    for i in range(0, Drug_sensi_df.shape[1]):
    # for i in range(0, 2):
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
            curr_test = pd.concat((test_features[layer_file_name], drug_labeled), axis=1, join='inner')

            #using "ElasticNet" tunned hyperparameters:
            C = hyperparameter_file.iloc[i,1]
            l1_ratio = hyperparameter_file.iloc[i,2]
            print(C,l1_ratio)
            model = LogisticRegression(penalty = 'elasticnet', solver = 'saga',C=C, l1_ratio = l1_ratio)
            model.fit(curr_train.iloc[:,:-1], curr_train.iloc[:, -1])
            train_pred_prob = model.predict_proba(curr_train.iloc[:,:-1])[:,1]
            test_pred_prob = model.predict_proba(curr_test.iloc[:,:-1])[:,1]
            # print("test_pred_prob",test_pred_prob)


            train_preds.loc[curr_train.index, drug] = train_pred_prob
            test_preds.loc[curr_test.index, drug] = test_pred_prob


            # if i == 0:
            #     with open(os.path.join(parameters_file,layer_file_name),'a') as f:
            #         writer = csv.writer(f)
            #         writer.writerow(["drug_id"] + hidden_rep_df.columns.to_list())
            # with open(os.path.join(parameters_file,layer_file_name),'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([drug] + list(model.coef_[0]))


            try:
                train_auroc = round(roc_auc_score(curr_train.iloc[:, -1],train_pred_prob),3)
            except:
                train_auroc = 0.5

            try:
                test_auroc = round(roc_auc_score(curr_test.iloc[:, -1], test_pred_prob),3)
            except:
                test_auroc = 0.5

            # print(train_f1, train_auroc, val_f1, val_auroc,test_f1, test_auroc)
            res_dict['train'][feature] = {'auroc':train_auroc}
            res_dict['test'][feature] = {'auroc':test_auroc}

        train_preds.to_csv(train_res_file.replace('train.csv', 'train_preds.csv'))
        test_preds.to_csv(test_res_file.replace('test.csv', 'test_preds.csv'))

        # write train results
        with open(train_res_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([drug,res_dict['train']['sga_hiddens']['auroc']])

        # write test results
        with open(test_res_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([drug,res_dict['test']['sga_hiddens']['auroc']])
