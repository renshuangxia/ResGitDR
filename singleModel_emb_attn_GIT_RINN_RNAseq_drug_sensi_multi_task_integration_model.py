import copy
import os, csv
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm
from math import sqrt
from sklearn.preprocessing import normalize, scale
from models.GIT_RINN_LuX_multi_task_integration_model import GIT_RINN

'''
    This script is to predict 367 drugs sensitivity and gene experission at the same time 
    using GIT-RINN model with input data of SGA, and it is using cross-validation
'''

'''
    print cpu information and set random seed
'''
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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
n_split = 5 # k-fold cross validation
num_epochs = 100
early_stop = 5

weight_decay = 0 # L2 norm
lamb = 1e-5 #L1 term, the value is 0 or 1e-5
alpha = 0.0 #drug sensivity part loss ratio
epoch_step_size = 5
alpha_increment = 0.1
maximum_alpha = 0.2

batch_size = 45
lr = 0.001  # learning rate
embedding_size = 200  # embedding sizes for GIT or number of nodes per hidden layer in NN
num_blocks = 4  # number of hidden layers or GIT blocks
n_head = 4
scale_y = True
using_cancer_type = True
using_RNAseq_type = "original_no_negative" # "QN"---quantile normalization or "original"---no normalization or "original_no_negative"

'''
    store result
'''
results_dir = "../result/sga_deg_result/results/single_model_drug_sensi_multi_task_GIT_RINN_drug_sensi_gep_" \
              +"cancertype_5fold_RNAseq_integration_model/"
def store_result_dir(results_dir, fold_idx, phrase):
    fold_dir = results_dir + "fold_" + str(fold_idx) + "/" + phrase + "/"
    attn_dir = fold_dir + 'attns/'
    emb_dir = fold_dir + 'embs/'
    hidden_weights_dir = fold_dir + 'hidden_weights/'
    res_dir = fold_dir + 'results/'
    hidden_dir = fold_dir + 'hidden/'
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    os.makedirs(fold_dir)
    os.mkdir(attn_dir)
    os.mkdir(emb_dir)
    os.mkdir(hidden_weights_dir)
    os.mkdir(res_dir)
    os.mkdir(hidden_dir)
    return {'attn_dir' : attn_dir, 'emb_dir' : emb_dir, "hidden_weights_dir":hidden_weights_dir,
            'hidden_dir': hidden_dir, 'res_dir' : res_dir}

'''
    The function of calculating the confidence interval of AUROC
'''
def roc_auc_ci(y_true, y_score, positive=1):
    try:
        AUC = roc_auc_score(y_true, y_score)
    except:
        AUC = 0.500
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    ci = round(1.96*SE_AUC,3)
    out = "{}".format(round(AUC,3))+"\u00B1"+"{}".format(ci)
    # lower = AUC - 1.96*SE_AUC
    # upper = AUC + 1.96*SE_AUC
    # if lower < 0:
    #     lower = 0
    # if upper > 1:
    #     upper = 1
    return out


'''
    Load GDSC data
'''
gdsc_sga_df = pd.read_csv("../data/gdsc_data/Cell_Model_Passports/TCGA_GDSC_RNAseq_sga_processed_data/GDSC_sga_data.csv",index_col=0)
gdsc_can_df = pd.read_csv("../data/gdsc_data/Cell_Model_Passports/TCGA_GDSC_RNAseq_sga_processed_data/GDSC_cancer_type.csv",index_col=0)
Drug_sensi_df = pd.read_csv("../data/gdsc_tcga_data_process_shuangxia/gdsc_processed_data/GDSC1_drug_response_binaryAUC.csv",index_col=0)
if using_RNAseq_type == "original":
    gdsc_RNAseq_df = pd.read_csv("../data/gdsc_data/Cell_Model_Passports/TCGA_GDSC_RNAseq_sga_processed_data/GDSC_RNAseq_data.csv",index_col=0)
if using_RNAseq_type == "QN":
    gdsc_RNAseq_df = pd.read_csv("../data/gdsc_data/Cell_Model_Passports/TCGA_GDSC_RNAseq_sga_processed_data/GDSC_RNAseq_QN_data.csv",index_col=0)
if using_RNAseq_type == "original_no_negative":
    gdsc_RNAseq_df = pd.read_csv("../data/gdsc_data/Cell_Model_Passports/TCGA_GDSC_RNAseq_sga_processed_data/GDSC_RNAseq_data.csv",index_col=0)
    gdsc_RNAseq_df[gdsc_RNAseq_df<0]=0
#set index to be the same
gdsc_sga_df.index = gdsc_sga_df.index.astype(str)
gdsc_can_df.index = gdsc_can_df.index.astype(str)
Drug_sensi_df.index = Drug_sensi_df.index.astype(str)
gdsc_RNAseq_df.index = gdsc_RNAseq_df.index.astype(str)
common_samples_list = [sample for sample in Drug_sensi_df.index if sample in gdsc_sga_df.index]

gdsc_sga_df = gdsc_sga_df.loc[common_samples_list,:]
gdsc_can_df = gdsc_can_df.loc[common_samples_list,:]
Drug_sensi_df = Drug_sensi_df.loc[common_samples_list,:]
gdsc_RNAseq_df = gdsc_RNAseq_df.loc[common_samples_list,:]

sga_df = gdsc_sga_df
can_df = gdsc_can_df
deg_df = gdsc_RNAseq_df


'''
    select genes for sga
'''
gene_sga_list = pd.read_csv("../data/gdsc_tcga_data_process_shuangxia/final_version_exprs_sga/TCI_fondation1_driver_dataset_combine_gene_list.txt",header=None)
common_gene_sga = list(set(sga_df.columns)&set(gene_sga_list.iloc[:,0].tolist()))
sga_df = sga_df.loc[:,common_gene_sga]
sga_source_df = sga_df #keep track of original index
# print("sga_source_df",sga_source_df)

'''
    select genes for deg
'''
gene_exprs_list = pd.read_csv("../data/gdsc_tcga_data_process_shuangxia/final_version_exprs_sga/exprs_gene_list_GDSC_yifan_mike.txt",header=None)
common_gene_exprs = list(set(deg_df.columns)&set(gene_exprs_list.iloc[:,0].tolist()))
deg_df = deg_df.loc[:,common_gene_exprs]

'''
    Map cancer type to indices for GIT models
'''
def map_cancer_type_to_id(can_df):
    can_idx = 0
    can2idx = {} # cancer type to index mapping
    for i in range(can_df.shape[0]):
        can_type = can_df.iloc[i, -1]
        if can_type in can2idx:
            can_df.iloc[i, -1] = can2idx[can_type]
        else:
            can2idx[can_type] = can_idx
            can_df.iloc[i, -1] = can_idx
            can_idx += 1
    return can_df,can_idx
can_df,can_size = map_cancer_type_to_id(can_df)
# print(can_df,can_size)

'''
    Preprocessing for GIT 
    Modify '1' in sga features to the sga indices, 0 remains 0 and will be using padding embedding
'''
def sga_features_to_indices(sga_df):
    sga_feats = sga_df.to_numpy()   #change 1/0 into the gene index
    non_zero_idx = np.where(sga_feats != 0)
    sga_feats[non_zero_idx[0], non_zero_idx[1]] = np.array(non_zero_idx[1]) + 1  #adding 1 since the zero index is perserved for padding embedding
    sga_df = pd.DataFrame(sga_feats, index=sga_df.index, columns=sga_df.columns) # reset values for sga df
    return sga_df
sga_df = sga_features_to_indices(sga_df)
# print(sga_df)


'''
    Train test split and data preparation
'''
X = sga_df
y = pd.concat([Drug_sensi_df,deg_df],axis=1)

kf = KFold(n_splits=n_split,shuffle=True)

for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):  # we do this at beginning so all training and tests are using same set
    alpha = 0 # reset alpha
    print("* Running fold {}".format(fold_idx))
    X_train = X.iloc[train_index]
    X_test = sga_source_df.iloc[test_index]

    can_train = can_df.loc[X_train.index].iloc[:, -1]  # cancer type indicies for training set
    can_test = can_df.loc[X_test.index].iloc[:, -1]  # cancer type indicies for testing set

    y_train = y.iloc[train_index]
    y_test =y.iloc[test_index]

    if scale_y == True:
        y_train.iloc[:,Drug_sensi_df.shape[1]:] = scale(y_train.iloc[:,Drug_sensi_df.shape[1]:], axis=1)
        y_test.iloc[:,Drug_sensi_df.shape[1]:] = scale(y_test.iloc[:,Drug_sensi_df.shape[1]:], axis=1)

    y_train = pd.DataFrame(y_train,index=X_train.index,columns=y.columns)
    y_test = pd.DataFrame(y_test,index=X_test.index,columns=y.columns)

    y_test_indices = y_test.index
    y_train_indices = y_train.index

    X_train = torch.LongTensor(X_train.values)
    X_test = torch.LongTensor(X_test.values)

    can_train = torch.LongTensor(can_train.values.astype(int)).unsqueeze(dim=1)
    can_test = torch.LongTensor(can_test.values.astype(int)).unsqueeze(dim=1)

    X_train = torch.cat((X_train, can_train), dim=1)
    X_test = torch.cat((X_test, can_test), dim=1)

    y_train = torch.FloatTensor(y_train.values)
    y_test = torch.FloatTensor(y_test.values)

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)

    # prepare data laoders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    data_loaders = {'train': train_loader, 'test': test_loader}


    '''
        Prepare models
    '''

    model: GIT_RINN = GIT_RINN(Drug_sensi_df.shape[1],deg_df.shape[1], sga_df.shape[1], embedding_size, can_size=can_size,
                               n_head=n_head, num_attn_blocks=num_blocks,using_cancer_type=using_cancer_type)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()


    '''
        Do training
    '''
    num_epoch_no_gain = 0
    min_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        # increment alpha when reaching step size on multi-task phase
        if alpha > 0 and alpha < maximum_alpha and (epoch - starting_step_epoch) % epoch_step_size == 0:
            alpha += alpha_increment
            if alpha >= maximum_alpha:
                alpha = maximum_alpha
                min_loss = float("inf") # reset min loss
                best_model = copy.deepcopy(model.state_dict())
                print("Start drug sensitivity single task learning phrase at epoch", epoch)
                print("reset min loss:", min_loss)

        if num_epoch_no_gain >= early_stop:
            if alpha == 0: # gene expression single task phrase complete, start multi-task phrase
                starting_step_epoch = epoch
                alpha += alpha_increment
                num_epoch_no_gain = 0 # reset no gain count
                model.load_state_dict(best_model)
                print("Complete single task gene expression phrase at epoch", epoch)
                print("Start multi-task learning phrase")
            else: # complete training
                print('Early stop at epoch:', epoch)
                break

        epoch_loss = 0.0
        model.train()
        print("epoch",epoch)
        num_batches = 0
        for X_batch, y_batch in tqdm(train_loader, disable=True, desc="Traininng Epoch"):  # for each batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output_drug,output_deg,_, _ = model(X_batch[:, 0:-1], X_batch[:, -1])
                y_batch_drug = y_batch[:,:Drug_sensi_df.shape[1]]
                y_batch_deg = y_batch[:,Drug_sensi_df.shape[1]:]

                sig_outputs_drug = torch.sigmoid(output_drug)
                # print(y_batch,sig_outputs)
                # print(y_batch[torch.isnan(y_batch)],sig_outputs[torch.isnan(y_batch)])
                y_batch_drug[torch.isnan(y_batch_drug)] = sig_outputs_drug[torch.isnan(y_batch_drug)]
                # print(y_batch)
                loss1 = criterion1(output_drug,y_batch_drug)
                loss2 = criterion2(output_deg,y_batch_deg)
                loss = alpha*loss1 + (1-alpha)*loss2

                l1_reg_loss = 0
                if alpha > 0.0 and lamb > 0: # do L1 regularization
                    # for param in model.hiddenLayers[-2].parameters():
                    l1_reg_loss += torch.norm(model.hiddenLayers[-2].weight,1)
                    l1_reg_loss = lamb * l1_reg_loss
                    loss += l1_reg_loss # add L1 regularization

                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
        epoch_loss /= num_batches
        print("*epoch_loss:",epoch_loss, "*alpha:", alpha, "*CELoss:", loss1.item(),
              "*MSELoss:", loss2.item(), "*L1Loss:", l1_reg_loss, "*loss:", loss.item())


        model.eval()
        test_epoch_loss = 0.0
        num_batches = 0

        #TODO: Test loss on phrase 3 should only consider cross entropy loss
        for X_batch, y_batch in tqdm(test_loader, disable=True):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                output_drug,output_deg, attns, hidden_outs = model(X_batch[:, 0:-1], X_batch[:, -1])
                y_batch_drug = y_batch[:,:Drug_sensi_df.shape[1]]
                y_batch_deg = y_batch[:,Drug_sensi_df.shape[1]:]

                sig_outputs_drug = torch.sigmoid(output_drug)
                y_batch_drug[torch.isnan(y_batch_drug)] = sig_outputs_drug[torch.isnan(y_batch_drug)]
                # print(y_batch)
                loss1 = criterion1(output_drug,y_batch_drug)
                loss2 = criterion2(output_deg,y_batch_deg)
                test_loss = alpha*loss1 + (1-alpha)*loss2
                print("test_phase:","l1_loss=",loss1,"loss2",loss2,"test_loss",test_loss )
                test_epoch_loss += loss1.item() if alpha > 0.0 else test_loss.item()
            num_batches += 1
        test_epoch_loss /= num_batches
        print("test_epoch_loss", test_epoch_loss)

        if alpha == 0.0 or alpha == maximum_alpha: # only do early stop at single task phrase
            if test_epoch_loss < min_loss:
                best_model = copy.deepcopy(model.state_dict())
                min_loss = test_epoch_loss
                num_epoch_no_gain = 0
            else:
                num_epoch_no_gain += 1
            scheduler.step(epoch_loss)

    model.load_state_dict(best_model)

    '''
        Evaluation
    '''
    model.eval()
    for phase in ['train', 'test']:
        loader = data_loaders[phase]
        pred_probs = []
        targets = []

        attn_wt_ls = []
        attn_wts_arrs = []

        hidden_out_list = []
        layer_out_arrs = []

        dirs_map = store_result_dir(results_dir,fold_idx,phase)

        # adding results and true labels
        for X_batch, y_batch in tqdm(loader, disable=True):
            X_batch = X_batch.to(device)
            # print(X_batch)
            with torch.no_grad():
                output_drug,output_deg, attns, hidden_outs = model(X_batch[:, 0:-1], X_batch[:, -1])
                output_drug = torch.sigmoid(output_drug)
                outputs = torch.cat((output_drug,output_deg),axis=1)
                # print("output",outputs.shape)

                attns = {i: attn.detach().cpu().numpy() for i, attn in attns.items()}
                hidden_outs = {i: hd_out.detach().cpu().numpy() for i, hd_out in hidden_outs.items()}
                pred_probs.append(outputs.cpu().detach().numpy())

                targets.append(y_batch)
                attn_wt_ls.append(attns)
                hidden_out_list.append(hidden_outs)

        pred_probs = np.concatenate(pred_probs)
        targets = np.concatenate(targets)

        '''
            Store attention weights
        '''
        for batch_attns in attn_wt_ls:
            # print(len(batch_attns), batch_attns[0])
            for idx, block_attn_wts in batch_attns.items():
                # print(idx, block_attn_wts)
                if idx >= len(attn_wts_arrs):
                    attn_wts_arrs.append(block_attn_wts)
                else:
                    attn_wts_arrs[idx] = np.concatenate((attn_wts_arrs[idx], block_attn_wts))

        for idx, attn_wts in enumerate(attn_wts_arrs):
            attn_df = pd.DataFrame(attn_wts, index=globals()["y_" + phase + "_indices"], columns=sga_df.columns)
            attn_df.to_csv(dirs_map["attn_dir"] +'Attn_Wts_' + str(idx) + '.csv')

        '''
            Store embeddings
        '''
        for idx, sga_block in enumerate(model.SGA_blocks):
            #print("sga paddings: ", sga_block.sga_embs.module.weight.data[0])
            sga_embs = sga_block.sga_embs.weight.data[1:].detach().cpu().numpy()
            sga_embs_df = pd.DataFrame(sga_embs, index=sga_df.columns)
            sga_embs_df = sga_embs_df.to_csv(dirs_map["emb_dir"] +'SGA_embs_' + str(idx) + '.csv')


        '''
            Store hidden layer weights
        '''
        for idx, hidden_layer in enumerate(model.hiddenLayers):
            #print("sga paddings: ", sga_block.sga_embs.module.weight.data[0])
            hidden_weights = hidden_layer.weight.data.detach().cpu().numpy()
            hidden_weights_df = pd.DataFrame(hidden_weights)
            hidden_weights_df.to_csv(dirs_map["hidden_weights_dir"] +'hidden_weights_' + str(idx) + '.csv')

        '''
            Store hidden representation
        '''
        for batch_res in hidden_out_list:
            for idx, block_res in batch_res.items():
                if idx >= len(layer_out_arrs):
                    layer_out_arrs.append(block_res)
                else:
                    layer_out_arrs[idx] = np.concatenate((layer_out_arrs[idx], block_res))
                # print(layer_out_arrs[idx].shape)

        for idx, layer_outs in enumerate(layer_out_arrs):
            attn_df = pd.DataFrame(layer_outs, index=globals()["y_" + phase + "_indices"])
            attn_df.to_csv(dirs_map["hidden_dir"] + '/hidden_outs_' + str(idx) + '.csv')

        '''
        Store predicts and targets
        '''
        predicts_df = pd.DataFrame(pred_probs, index=globals()["y_" + phase + "_indices"], columns=y.columns)
        targets_df = pd.DataFrame(targets, index=globals()["y_" + phase + "_indices"], columns=y.columns)

        predicts_df.to_csv(dirs_map["res_dir"] + "predicts.csv")
        targets_df.to_csv(dirs_map["res_dir"] + "targets.csv")