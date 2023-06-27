import torch.nn as nn
import numpy as np
import torch
from gensim.models import Word2Vec
import pandas as pd



class SGASelfAttn(nn.Module):
    """
    This class implement self-attention mechanism for SGA.  Given a vector of SGA status of m genes, this class produced
    a vector of embedding_dim, which corresponding to the dimension of vector embedding representing each gene.  Current
    implementation support multiple heads self-attention.
    """

    def __init__(self,sga_gene_list, sga_size, embedding_dim,n_head,using_gene2vec_pretrain):
        """ Initialize the model.
        Parameters:
                      sga_size          The total number of SGA, the number of sga embeddings to create (plus 1 for padding)
                      embedding_dim     The dimension of embedding vector for each gene
                      n_head            The number of head in self-attention mechanism
        """
        super(SGASelfAttn, self).__init__()
        self.sga_gene_list = sga_gene_list
        self.sga_size = sga_size
        self.embeddingDim = embedding_dim
        self.attention_head = n_head
        self.attention_size = embedding_dim
        self.using_gene2vec_pretrain = using_gene2vec_pretrain

        # set up a look up embedding matrix for SGAs, with first embedding vector being "0" at indx 0 as padding
        # A vector in this matrix corresponding to input vecotr (look up) as well as the v (identify function)

        if using_gene2vec_pretrain == True:
            print("using_gene2vec_pretrain")
            gene2vec_model = Word2Vec.load("data/gene2vec/word2vec.model_"+str(embedding_dim))
            w2v_dict = {item : gene2vec_model.wv[item] for item in gene2vec_model.wv.key_to_index}
            gene2vec_df = pd.DataFrame.from_dict(w2v_dict)
            gene2vec_df = gene2vec_df.loc[:,self.sga_gene_list]
            embedding_matrix = np.insert(gene2vec_df.values,0,np.zeros(gene2vec_df.shape[0]),axis=1)
            self.sga_embs = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix).T, padding_idx=0,freeze=False)
        else:
            self.sga_embs = nn.Embedding(num_embeddings=self.sga_size + 1, embedding_dim=int(self.embeddingDim),
                                         padding_idx=0)


        # Set up attention mechanism
        # First project input data to produce key vectors for each SGA.
        self.transform2Keys = nn.Linear(in_features=self.embeddingDim, out_features=self.embeddingDim, bias=True)

        # Input for the project is m-by-embedding_dim, output is a n-by-m
        self.queryKeyMatch = nn.Linear(in_features=self.embeddingDim, out_features=self.attention_head, bias=True)

    def forward(self, sga, mask=True):
        """
        parameters:   sga   A matrix represent SGAs in cases. Each row corresponding to a cases, and
                            a column correspond to a gene.  A "0" in a cell i,j indicates gene j in case i is normal.
                            We indicate an altered gene in gene j using an indix number corresponding to j,
                            so that the program can pick correct embedding for gene j in case i

        Dimension: B - batch size ,
                   I - input dim, number of gene indices selected for each cell line
                   E - embedding dimension
                   A - attention dimension
                   N - number of attention heads
        """

        # Look up gene embedings based on input SGA data. Produce a m-by-embedding_dim matrix
        case_sga_embs = self.sga_embs(sga)  # B x I x E (e.g. 100 x 300 x 128)
        # print(case_sga_embs.shape)

        # transform m-by-embedding_dim data matrix through a tanh transformation, out put is again a m-by-embedding matrix
        case_sga_keys = torch.tanh(self.transform2Keys(case_sga_embs))  # B x I x A (e.g. 100 x 300 x 128)

        # multiplied by query matrix Wq to derive multi-head attention, an n-by-m matrix
        case_sga_attn_heads = self.queryKeyMatch(case_sga_keys)  # B X I X N (e.g. 100 x 300 x 5)
        
        if mask == True:
            sga_mask = sga.repeat(self.attention_head,1,1).permute(1,2,0)
            case_sga_attn_heads = case_sga_attn_heads.masked_fill(sga_mask ==0,-1e9)

        A = torch.softmax(case_sga_attn_heads, dim=1)  # B X I X N (e.g. 100 x 300 x 5)

        # multi-head attention weighted cancer embedding:
        emb_sga = torch.sum(torch.bmm(A.permute(0, 2, 1), case_sga_embs), dim=1)  # B X E (e.g. 100 x 128)

        emb_signal = emb_sga

        # attention weights
        attn_wt = torch.sum(A, dim=2)  # B x I (e.g. 100 x 300)

        return emb_signal, attn_wt


class ResGit(nn.Module):


  def __init__(self, out_feat_size, sga_size, embedding_dim, embedding_dim_last, can_size=10, n_head=5,
               num_attn_blocks=3,using_cancer_type=True,using_tf_gene_matrix=False,tf_gene=None,sga_gene_list=None,
               embed_pretrain_gene2vec=True):
    super(ResGit, self).__init__()
    self.sga_size = sga_size  #number of sga genes in total
    self.out_feat_size = out_feat_size #output embedding dimension
    self.can_size = can_size #number of cancer types in total
    self.embedding_dim = embedding_dim # embedding dimension
    self.embedding_dim_last = embedding_dim_last # embedding dimension for last layer
    self.attention_head = n_head #number of self attention heads
    self.activationF = nn.ReLU()
    self.num_attn_blocks = num_attn_blocks # number of sga attention blocks
    self.using_cancer_type = using_cancer_type #whether using cancer type information
    self.using_tf_gene_matrix = using_tf_gene_matrix #whether using tf-gene matrix
    self.tf_gene = tf_gene # the tf-gene matrix
    self.sga_gene_list = sga_gene_list #the sga gene list
    self.embed_pretrain_gene2vec = embed_pretrain_gene2vec #whether using gene2vec to pretrain the gene embedding

    '''
        sga and can embeddings
    '''
    # cancer type embedding is a one-hot-vector, no need to pad
    self.layer_can_emb = nn.Embedding(num_embeddings=self.can_size, embedding_dim=self.embedding_dim)

    '''
        Construct the architecture of the GIT_RINN. 
    '''
    # we use two module lists: one to keep track of feed forward chain of hidden layers,
    # and one to keep track from SGA to hidden
    self.SGA_blocks = nn.ModuleList()
    self.hiddenLayers = nn.ModuleList()

    # First block is from SGA to hidden 0
    self.SGA_blocks.append(SGASelfAttn(self.sga_gene_list,self.sga_size, self.embedding_dim, self.attention_head,
                                       self.embed_pretrain_gene2vec))

    # populate the structure of hidden layers
    for i in range(1, num_attn_blocks-1):
        self.SGA_blocks.append(SGASelfAttn(self.sga_gene_list,self.sga_size, self.embedding_dim,self.attention_head,
                                           self.embed_pretrain_gene2vec))
        linearlayer = nn.Linear(self.embedding_dim, self.embedding_dim,bias=False)
        self.hiddenLayers.append(linearlayer)

    self.SGA_blocks.append(SGASelfAttn(self.sga_gene_list,self.sga_size, self.embedding_dim_last,self.attention_head,
                                       self.embed_pretrain_gene2vec))
    linearlayerLast = nn.Linear(self.embedding_dim, self.embedding_dim_last,bias=False)
    self.hiddenLayers.append(linearlayerLast)

    # final hidden to output layer
    if using_tf_gene_matrix:
        self.layer_final = nn.Linear(self.tf_gene.shape[0], self.out_feat_size,bias=False)
        # mask_value = torch.FloatTensor(self.tf_gene.T)
        self.mask_value = self.tf_gene.T

        # define layer weight clapped by mask
        self.layer_final.weight.data = self.layer_final.weight.data * torch.FloatTensor(self.tf_gene.T)
        # register a backford hook with the mask value
        self.layer_final.weight.register_hook(lambda grad: grad.mul_(self.mask_value))
        self.hiddenLayers.append(self.layer_final)
    else:
        self.hiddenLayers.append(nn.Linear(self.embedding_dim_last, self.out_feat_size,bias=False))


  def forward(self, sga_index, can_index, store_hidden_layers=True):
    """ Forward process.
      Parameters
      ----------
      sga_index: list of sga index vectors.
      can_index: list of cancer type indices.
      -------
    """
    # attn_wts = [] # stores attention weights

    # First, feed forward from SGA to hidden 0
    curr_hidden, attn_wts = self.SGA_blocks[0](sga_index)
    if self.using_cancer_type:
        curr_hidden = curr_hidden + self.layer_can_emb(can_index)
    curr_hidden = self.activationF(curr_hidden)

    batch_attn_wts = {}
    batch_attn_wts[0] = attn_wts # add attention weights obtained from first sga attention block

    hidden_layer_outs = {}
    hidden_layer_outs[0] = curr_hidden


    for i in range(1, len(self.hiddenLayers)):
        sga_emb_after_attn, attn_wts = self.SGA_blocks[i](sga_index)
        curr_hidden = sga_emb_after_attn + self.hiddenLayers[i-1](curr_hidden)
        curr_hidden = self.activationF(curr_hidden) if i < len(self.hiddenLayers) - 1 else torch.sigmoid(curr_hidden)
        # attn_wts = attn_wts.detach().cpu().numpy() # store weights in numpy array
        if store_hidden_layers:
            hidden_layer_outs[i] = curr_hidden

        # add to attention weigths dictionary, key i represents i'th sga attention block
        batch_attn_wts[i] = attn_wts

    preds = self.hiddenLayers[-1](curr_hidden)
    return preds, batch_attn_wts, hidden_layer_outs


    

