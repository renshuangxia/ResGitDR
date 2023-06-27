import copy
import math
import os, csv
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.preprocessing import normalize, scale
from models.ResGit import ResGit
import torch.nn.utils.prune as prune

'''
    This script is to train ResGit using 10 fold cross-validation, the input is SGA and cancer type, 
    the output is gene expression values.
'''


'''
    print cpu information and set random seed
'''
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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
n_split = 10 # k-fold cross validation
num_epochs = 60
weight_decay = 0 # L2 norm
hidden_weights_pruning = True #True or False, if True pruning the hidden weights
hidden_weights_pruning_ratio = 0.9
early_stop = 50

batch_size = 45
lr = 0.001  # learning rate
embedding_size = 200  # embedding sizes for GIT or number of nodes per hidden layer in NN
embedding_size_last = 320 # embedding sizes or number of nodes in the last hidden layer
num_blocks = 4  # number of hidden layers or GIT blocks
n_head = 4
using_cancer_type = True
scale_y = True #scale target in case the gradient dominated by larger numbers
using_tf_gene_matrix = True

trim_ratio = 0.6  #pruning ratio for embedding
top_k = math.floor(embedding_size * trim_ratio)
top_k_last_layer = math.floor(embedding_size_last * trim_ratio)

embed_pretrain_gene2vec = True
permutation_sga_tissue = False #permutating the cancer type and random generate the sga value and cancer type according to the distribution in the training dataset for each gene and cancer type"


'''
    store result
'''
results_dir = "result/cross_validation/ResGit/"
def store_result_dir(results_dir, fold_idx, phrase):
    fold_dir = results_dir + "fold_" + str(fold_idx) + "/" + phrase + "/"
    res_dir = fold_dir + 'results/'
    hidden_dir = fold_dir + 'hidden/'
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    os.makedirs(fold_dir)
    os.mkdir(res_dir)
    os.mkdir(hidden_dir)
    return {'hidden_dir': hidden_dir, 'res_dir' : res_dir}

''' 
    permutate the cancer type: according to the multinomial distribution of different types of cancers
'''
def cancer_type_permutation(train_cancer_df,test_cancer_df):
    train_cancer_type_multinomial = train_cancer_df.value_counts()/train_cancer_df.shape[0]
    cancer_type_permute_list = np.random.choice(train_cancer_type_multinomial.index, test_cancer_df.shape[0], p=train_cancer_type_multinomial.values)
    test_cancer_df = test_cancer_df.to_frame()
    test_cancer_df["permutation_type"] = cancer_type_permute_list

    return test_cancer_df["permutation_type"]

'''
    Load TF-GENE matrix
'''
if using_tf_gene_matrix == True:
    tf_gene_df = pd.read_csv("data/CITRUS_tf_gene.csv",index_col=0)
    tf_gene_df[tf_gene_df>0]=1
    tf_gene_gene_list = tf_gene_df.columns


'''
    Load TCGA data
'''
tcga_sga_df = pd.read_csv("data/TCGA_sga_data.csv",index_col=0)
tcga_RNAseq_df = pd.read_csv("data/TCGA_RNAseq_data.csv",index_col=0)
tcga_RNAseq_df[tcga_RNAseq_df<0]=0
tcga_can_df = pd.read_csv("data/TCGA_xena_map_cancertype.csv",index_col=0)

tcga_RNAseq_df = tcga_RNAseq_df.loc[tcga_sga_df.index,:]
tcga_can_df = tcga_can_df.loc[tcga_sga_df.index,["cancer_type"]]

'''
    Load GDSC data
'''
gdsc_sga_df = pd.read_csv("data/GDSC_sga_data.csv",index_col=0)
gdsc_RNAseq_df = pd.read_csv("data/GDSC_RNAseq_data.csv",index_col=0)
gdsc_RNAseq_df[gdsc_RNAseq_df<0]=0
gdsc_can_df = pd.read_csv("data/GDSC_cancer_type.csv",index_col=0)

gdsc_RNAseq_df = gdsc_RNAseq_df.loc[gdsc_sga_df.index,:]
gdsc_can_df = gdsc_can_df.loc[gdsc_sga_df.index,:]

'''
    Combine TCGA and GDSC data
'''
sga_df = pd.concat([tcga_sga_df,gdsc_sga_df])
deg_df = pd.concat([tcga_RNAseq_df,gdsc_RNAseq_df])
can_df = pd.concat([tcga_can_df,gdsc_can_df])


'''
    select genes for sga and deg
'''
gene_exprs_list = pd.read_csv("data/exprs_gene_list_GDSC_yifan_mike.txt",header=None)
gene_sga_list = pd.read_csv("data/TCI_fondation1_driver_dataset_combine_gene_list.txt",header=None)

common_gene_sga = [sga_gene for sga_gene in sga_df.columns if sga_gene in gene_sga_list.iloc[:,0].tolist()]
common_gene_exprs = [deg_gene for deg_gene in deg_df.columns if deg_gene in gene_exprs_list.iloc[:,0].tolist()]

if using_tf_gene_matrix == True:
    common_gene_exprs = [gene for gene in common_gene_exprs if gene in tf_gene_gene_list]
    tf_gene_df = tf_gene_df.loc[:,common_gene_exprs]

sga_df = sga_df.loc[:,common_gene_sga]
deg_df = deg_df.loc[:,common_gene_exprs]
sga_source_df = sga_df.copy() #keep track of original index

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
print(can_df,can_size)


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
print(sga_df,deg_df)


'''
    Train test split and data preparation
'''
X = sga_df
y = deg_df

kf = KFold(n_splits=n_split,shuffle=True)


'''
    use both GDSC and TCGA data
'''
for fold_idx, ((tcga_train_index, tcga_test_index), (gdsc_train_index, gdsc_test_index)) in \
        enumerate(zip(kf.split(X.iloc[0:tcga_sga_df.shape[0],:]),kf.split(X.iloc[tcga_sga_df.shape[0]:,:]))):
    #since appling on kfold function, the index will from 0, but in our dataset, we concat gdsc after tcga

    tcga_train_index = tcga_train_index.tolist()
    tcga_test_index = tcga_test_index.tolist()
    gdsc_train_index = [i+tcga_sga_df.shape[0] for i in gdsc_train_index]
    gdsc_test_index = [j+tcga_sga_df.shape[0] for j in gdsc_test_index]

    # we do this at beginning so all training and tests are using same set
    print("* Running fold {}".format(fold_idx))

    X_train = X.iloc[tcga_train_index+gdsc_train_index,:]
    X_test = X.iloc[tcga_test_index+gdsc_test_index,:]

    can_train = can_df.loc[X_train.index].iloc[:, -1]  # cancer type indicies for training set
    can_test = can_df.loc[X_test.index].iloc[:, -1]  # cancer type indicies for testing set

    #if used the permuted data, then generated the permuted data according the distribution of traning data
    if permutation_sga_tissue == True:
        for gene in X_test.columns:
            each_gene_benurli = X_train[gene].value_counts()/X_train.shape[0]
            X_test[gene] = np.random.choice(each_gene_benurli.index, X_test.shape[0], p=each_gene_benurli.values)
            can_test = cancer_type_permutation(can_train,can_test)

    y_train = y.loc[X_train.index]
    y_test =y.loc[X_test.index]

    if scale_y == True:
        y_train = scale(y_train, axis=1)
        y_test = scale(y_test, axis=1)

    y_train = pd.DataFrame(y_train,index=X_train.index,columns=deg_df.columns)
    y_test = pd.DataFrame(y_test,index=X_test.index,columns=deg_df.columns)

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

    tf_gene = torch.FloatTensor(tf_gene_df.values) if using_tf_gene_matrix else None

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)

    # prepare data laoders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    data_loaders = {'train': train_loader, 'test': test_loader}


    '''
        Prepare models
    '''
    model: ResGit = ResGit(y_train.shape[1], sga_df.shape[1], embedding_size, embedding_size_last,can_size=can_size,
                             n_head=n_head, num_attn_blocks=num_blocks,using_cancer_type=using_cancer_type,
                               using_tf_gene_matrix=using_tf_gene_matrix,tf_gene=tf_gene,
                               sga_gene_list=sga_df.columns, embed_pretrain_gene2vec=embed_pretrain_gene2vec)
    model = model.to(device)
    if using_tf_gene_matrix:
        model.mask_value = model.mask_value.to(device)

    if hidden_weights_pruning == True:
        for ind,(name, module) in enumerate(model.hiddenLayers.named_modules()):
            if isinstance(module, torch.nn.Linear):
                if ind == len(model.hiddenLayers):
                    break    #the last layer---the TF layer doesn't apply the pruning
                # print(name,module)
                prune.l1_unstructured(module, name='weight', amount=hidden_weights_pruning_ratio)


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()

    '''
        Do training
    '''

    num_epoch_no_gain = 0
    min_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())

    complete_emb_pretrain = False
    for epoch in range(num_epochs):
        print(num_epoch_no_gain, early_stop)
        if trim_ratio != 0:
            trim_starting_epoch = 30
        else:
            trim_starting_epoch = num_epochs
        if num_epoch_no_gain > early_stop or epoch == trim_starting_epoch:
            # after 30 epochs or model converge, only using the trimmed embedding
            if complete_emb_pretrain:
                print('Early stop at epoch:', epoch)
                break
            #trim embedding
            else:
                model.load_state_dict(best_model)
                min_loss = float('inf')

                complete_emb_pretrain = True
                print("Finish embedding pretrain")

                for idx,sga_block in enumerate(model.SGA_blocks):
                    # print("Before:", sga_block.sga_embs.weight)
                    sga_block.sga_embs.weight.requires_grad = False
                    embs_wt = sga_block.sga_embs.weight
                    abs_wt = torch.abs(embs_wt)
                    if idx == num_blocks-1:
                        for t in range(abs_wt.shape[0]):
                            single_emb_wt = abs_wt[t]
                            thres, _ = torch.kthvalue(single_emb_wt, top_k_last_layer)
                            embs_wt[t, single_emb_wt < thres] = 0.0

                        embs_wt_df = pd.DataFrame(embs_wt.detach().cpu().numpy(), index=["padding"]+sga_df.columns.to_list(),columns=tf_gene_df.index)
                        common_gene_tf_sga = list(set(embs_wt_df.index)&set(embs_wt_df.columns))
                        print(common_gene_tf_sga )
                        for gene in common_gene_tf_sga:
                            embs_wt_df.loc[gene,gene]=1
                        print(embs_wt_df)
                        embs_wt = torch.nn.Parameter(torch.Tensor(embs_wt_df.values).to(device),requires_grad=False)
                    else:
                        for t in range(abs_wt.shape[0]):
                            single_emb_wt = abs_wt[t]
                            thres, _ = torch.kthvalue(single_emb_wt, top_k)
                            embs_wt[t, single_emb_wt < thres] = 0.0
                    sga_block.sga_embs.weight = embs_wt
                    # print("After:", sga_block.sga_embs.weight)
                num_epoch_no_gain = 0 # reset num epoch no gain
                optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad],
                                       lr=lr,
                                       weight_decay=weight_decay)

        epoch_loss = 0.0
        model.train()
        print("epoch",epoch)
        num_batches = 0

        for X_batch, y_batch in tqdm(train_loader, disable=True, desc="Traininng Epoch"):  # for each batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                #hidden3-to TF,
                outputs, _, _ = model(X_batch[:, 0:-1], X_batch[:, -1], False)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
        epoch_loss /= num_batches
        print("epoch_loss",epoch_loss)


        model.eval()
        test_epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in tqdm(test_loader, disable=True):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                outputs, attns, hidden_outs = model(X_batch[:, 0:-1], X_batch[:, -1], False)
                test_loss = criterion(outputs, y_batch)
                test_epoch_loss += test_loss.item()
            num_batches += 1
        test_epoch_loss /= num_batches
        print("test_epoch_loss",test_epoch_loss)

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
                outputs, attns, hidden_outs = model(X_batch[:, 0:-1], X_batch[:, -1], )
                attns = {i: attn.detach().cpu().numpy() for i, attn in attns.items()}
                hidden_outs = {i: hd_out.detach().cpu().numpy() for i, hd_out in hidden_outs.items()}
                pred_probs.append(outputs.cpu().detach().numpy())

                targets.append(y_batch)
                attn_wt_ls.append(attns)
                hidden_out_list.append(hidden_outs)

        pred_probs = np.concatenate(pred_probs)
        targets = np.concatenate(targets)

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
        predicts_df = pd.DataFrame(pred_probs, index=globals()["y_" + phase + "_indices"], columns=deg_df.columns)
        targets_df = pd.DataFrame(targets, index=globals()["y_" + phase + "_indices"], columns=deg_df.columns)

        predicts_df.to_csv(dirs_map["res_dir"] + "predicts.csv")
        targets_df.to_csv(dirs_map["res_dir"] + "targets.csv")