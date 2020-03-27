# import

from __future__ import print_function
from __future__ import division
import numpy as np
import networkx as nx
import scipy.sparse as sp
import time
import os
import pickle
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import tensorflow as tf
import sys
import argparse


# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


# Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--hd', type=int, default=32, help='Number of hidden units for DD-VGAE.')
parser.add_argument('--zd', type=int, default=16, help='Number of latent dimensions for DD-VGAE.')
parser.add_argument('--hg', type=int, default=64, help='Number of hidden units for GG-VGAE.')
parser.add_argument('--zg', type=int, default=32, help='Number of latent dimensions for GG-VGAE.')
parser.add_argument('--nfeats', type=int, default=32, help='Dimension of GG features after transform.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()


# helper functions borrowed 

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj, num_false):
    edges_positive, _, _ = sparse_to_tuple(adj)
    # filtering out edges from lower triangle of adjacency matrix
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:]
    test_edges_false = None

    # number of positive (and negative) edges in test and val sets:
    num_positive = np.floor(edges_positive.shape[0])
    num_max_false = np.floor(adj.shape[0]**2 / 2) - num_positive
    num_test = int(num_false)
    if num_false < num_max_false and 0.3*np.floor(adj.shape[0]**2 / 2)>num_positive:
        num_test = int(num_positive)
    
    print(num_test)
    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices

    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < num_test:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test-len(test_edges_false)), replace=True)
        idx = idx[~np.in1d(idx,positive_idx,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_test_edges_false,assume_unique=True)]
        
        if len(list(idx))==0:
            continue

        
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        coords = np.unique(coords,axis=0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        coords = coords[coords[:,0]!=coords[:,1]]
        coords = coords[:min(num_test,len(idx))]
        test_edges_false = np.append(test_edges_false,coords,axis=0)
        idx = idx[:min(num_test,len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    # sanity checks:
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    
    return edges_positive, test_edges_false


def get_roc_score(edges_pos, edges_neg, emb, adj_in):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_in[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_in[e[0], e[1]])
    
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score




# load data

adj_d = np.load('./data/disease_adj.npy')
adj_g = np.load('./data/genes_genes_net.npy')
att0 = np.load('./data/gene_GO1_net.npy')
att1 = np.load('./data/gene_GO2_net.npy')
att2 = np.load('./data/gene_GO3_net.npy')
att3 = np.load('./data/gene_pathway.npy')
att4 = np.load('./data/disease_gene_net.npy')



# putting adjs in sparse format

# adj_d = sp.csr_matrix(adj_d.astype(np.int64))
adjs_g = [sp.csr_matrix(adj_g[:,:,i].astype(np.int64)) for i in range(adj_g.shape[-1])]
att0 = sp.lil_matrix(att0.astype(np.float32))
att1 = sp.lil_matrix(att1.astype(np.float32))
att2 = sp.lil_matrix(att2.astype(np.float32))
att3 = sp.lil_matrix(att3.astype(np.float32))
att4 = sp.lil_matrix(np.transpose(att4.astype(np.float32)))



# preprocessing gene-gene network

num_nodes_g = adjs_g[0].shape[0]
adjs_g_orig = []
adjs_g_train = []
adjs_g_norm = []

for i in range(len(adjs_g)):
    adj_orig_temp = copy.deepcopy(adjs_g[i])
    adj_orig_temp = adj_orig_temp - sp.dia_matrix((adj_orig_temp.diagonal()[np.newaxis, :], [0])
                                                  , shape=adj_orig_temp.shape)
    adj_orig_temp.eliminate_zeros()
    adjs_g_orig.append(adj_orig_temp)

    adjs_g_train.append(copy.deepcopy(adjs_g_orig[-1]))
    adjs_g_norm.append(preprocess_graph(adjs_g_train[-1]))


ndatts0 = sparse_to_tuple(att0.tocoo())
num_ndatts0 = ndatts0[2][1]
ndatts_nonzero0 = ndatts0[1].shape[0]

ndatts1 = sparse_to_tuple(att1.tocoo())
num_ndatts1 = ndatts1[2][1]
ndatts_nonzero1 = ndatts1[1].shape[0]

ndatts2 = sparse_to_tuple(att2.tocoo())
num_ndatts2 = ndatts2[2][1]
ndatts_nonzero2 = ndatts2[1].shape[0]

ndatts3 = sparse_to_tuple(att3.tocoo())
num_ndatts3 = ndatts3[2][1]
ndatts_nonzero3 = ndatts3[1].shape[0]

ndatts4 = sparse_to_tuple(att4.tocoo())
num_ndatts4 = ndatts4[2][1]
ndatts_nonzero4 = ndatts4[1].shape[0]



# preprocessing disease-disease network

num_nodes_d = adj_d.shape[0]
adj_d_orig =  1.0 * (adj_d<0)
adj_d_orig = sp.csr_matrix(adj_d_orig.astype(np.int64))
adj_d_orig = adj_d_orig - sp.dia_matrix((adj_d_orig.diagonal()[np.newaxis, :], [0]), shape=adj_d_orig.shape)
adj_d_orig.eliminate_zeros()

adj_d_train = copy.deepcopy(adj_d_orig)
adj_d_norm = preprocess_graph(adj_d_train)

adj_d_label = adj_d_train + sp.eye(adj_d_train.shape[0])
adj_d_label = sparse_to_tuple(adj_d_label)



# creating gene-gene aggregated adjacency matrix

adj_g_label = sp.csr_matrix((1.0*(adjs_g_orig[0]+adjs_g_orig[1]+adjs_g_orig[2]+adjs_g_orig[3]
                                  +adjs_g_orig[4]).todense().astype(np.bool)).astype(np.int64))

adj_g_label = adj_g_label + sp.eye(adj_g_label.shape[0])
adj_g_label = sparse_to_tuple(adj_g_label)




# sampling negative edges for ROC/AP

ep_d, en_d = mask_test_edges(adj_d_orig, 5000)
adj_g_recon = sp.csr_matrix((1.0*(adjs_g_orig[0]+adjs_g_orig[1]+adjs_g_orig[2]+adjs_g_orig[3]
                                  +adjs_g_orig[4]).todense().astype(np.bool)).astype(np.int64))

ep_g, en_g = mask_test_edges(adj_g_recon, 5000)



tf.reset_default_graph()


# convolutional layers and decoer

_LAYER_UIDS = {}
slim = tf.contrib.slim

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionK(Layer):
    """Graph convolution layer for a graph with k edge types."""
    def __init__(self, input_dim, output_dim, adjs, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionK, self).__init__(**kwargs)
        self.k = len(adjs)
        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.k):
                self.vars['weights'+str(i)] = weight_variable_glorot(input_dim, output_dim
                                                                     , name="weights"+str(i))
        self.dropout = dropout
        self.adjs = adjs
        self.act = act
        self.output_dim = output_dim

    def _call(self, inputs):
        for i in range(self.k):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'+str(i)])
            x = tf.sparse_tensor_dense_matmul(self.adjs[i], x)
            if i == 0:
                outputs = tf.expand_dims(self.act(x), axis=-1)
            else:
                outputs = tf.concat([outputs, tf.expand_dims(self.act(x), axis=-1)], axis=-1)
        
        with tf.variable_scope(self.name + '_vars'):
            outputs = slim.fully_connected(outputs, self.k, activation_fn=tf.nn.relu)
            outputs = slim.fully_connected(outputs, self.k, activation_fn=tf.nn.relu)
            outputs = tf.squeeze(slim.fully_connected(outputs, 1, activation_fn=None))
        return outputs


class GraphConvolutionSparseK(Layer):
    """Graph convolution layer for a graph with k edge types and sparse inputs."""
    def __init__(self, input_dim, output_dim, adjs, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseK, self).__init__(**kwargs)
        self.k = len(adjs)
        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.k):
                self.vars['weights'+str(i)] = weight_variable_glorot(input_dim, output_dim
                                                                     , name="weights"+str(i))
        self.dropout = dropout
        self.adjs = adjs
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        for i in range(self.k):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'+str(i)])
            x = tf.sparse_tensor_dense_matmul(self.adjs[i], x)
            if i == 0:
                outputs = tf.expand_dims(self.act(x), axis=-1)
            else:
                outputs = tf.concat([outputs, tf.expand_dims(self.act(x), axis=-1)], axis=-1)
        
        with tf.variable_scope(self.name + '_vars'):
            outputs = slim.fully_connected(outputs, self.k, activation_fn=tf.nn.relu)
            outputs = slim.fully_connected(outputs, self.k, activation_fn=tf.nn.relu)
            outputs = tf.squeeze(slim.fully_connected(outputs, 1, activation_fn=None))
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelVAE(Model):
    """VGAE encoder for a graph."""
    def __init__(self, placeholders, num_features, num_nodes, nhids, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.nhids = nhids
        self.build()
        
    def _build(self):
        self.hidden1 = GraphConvolution(input_dim=self.input_dim
                                        ,output_dim=self.nhids[0]
                                        ,adj=self.adj
                                        ,act=tf.nn.relu
                                        ,dropout=self.dropout
                                        ,name='hid'
                                        ,logging=self.logging)(self.inputs)
        
        self.z_mean = GraphConvolution(input_dim=self.nhids[0]
                                       ,output_dim=self.nhids[1]
                                       ,adj=self.adj
                                       ,act=lambda x: x
                                       ,dropout=self.dropout
                                       ,name='mean'
                                       ,logging=self.logging)(self.hidden1)
        
        self.z_log_std = GraphConvolution(input_dim=self.nhids[0]
                                          ,output_dim=self.nhids[1]
                                          ,adj=self.adj
                                          ,act=lambda x: x
                                          ,dropout=self.dropout
                                          ,name='std'
                                          ,logging=self.logging)(self.hidden1)
        
        self.z = self.z_mean + tf.random_normal([self.n_samples, self.nhids[1]]) * tf.exp(self.z_log_std)
        
        self.reconstructions = InnerProductDecoder(input_dim=self.nhids[1]
                                                   ,act=lambda x: x
                                                   ,logging=self.logging)(self.z)


class GCNModelVAEK(Model):
    """VGAE encoder for a graph with k edge types."""
    def __init__(self, placeholders, num_features, num_nodes, nhids, **kwargs):
        super(GCNModelVAEK, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adjs = placeholders['adjs']
        self.dropout = placeholders['dropout']
        self.nhids = nhids
        self.build()
        
    def _build(self):
        self.hidden1 = GraphConvolutionK(input_dim=self.input_dim
                                        ,output_dim=self.nhids[0]
                                        ,adjs=self.adjs
                                        ,act=tf.nn.relu
                                        ,dropout=self.dropout
                                        ,name='hid'
                                        ,logging=self.logging)(self.inputs)
        
        self.z_mean = GraphConvolutionK(input_dim=self.nhids[0]
                                        ,output_dim=self.nhids[1]
                                        ,adjs=self.adjs
                                        ,act=lambda x: x
                                        ,dropout=self.dropout
                                        ,name='mean'
                                        ,logging=self.logging)(self.hidden1)
        
        self.z_log_std = GraphConvolutionK(input_dim=self.nhids[0]
                                           ,output_dim=self.nhids[1]
                                           ,adjs=self.adjs
                                           ,act=lambda x: x
                                           ,dropout=self.dropout
                                           ,name='std'
                                           ,logging=self.logging)(self.hidden1)
        
        self.z = self.z_mean + tf.random_normal([self.n_samples, self.nhids[1]]) * tf.exp(self.z_log_std)
        
        self.reconstructions = InnerProductDecoder(input_dim=self.nhids[1]
                                                   ,act=lambda x: x
                                                   ,logging=self.logging)(self.z)



class HierEmb(Model):
    """HVGAE model."""
    def __init__(self, plc_list, nfeats_in_list, nfeats_out, nnodes_list, feats_nz_list
                 , feats_list, nhids_list, dropout, **kwargs):
        super(HierEmb, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.adjs = plc_list[0]['adjs']
        
        self.feat0 = GraphConvolutionSparseK(input_dim=nfeats_in_list[0]
                                               ,output_dim=nfeats_out
                                               ,adjs=self.adjs
                                               ,features_nonzero=feats_nz_list[0]
                                               ,act=tf.nn.relu
                                               ,dropout=self.dropout
                                               ,name='sp0'
                                               ,logging=self.logging)(plc_list[0]['feats'][0])
        
        self.feat1 = GraphConvolutionSparseK(input_dim=nfeats_in_list[1]
                                               ,output_dim=nfeats_out
                                               ,adjs=self.adjs
                                               ,features_nonzero=feats_nz_list[1]
                                               ,act=tf.nn.relu
                                               ,dropout=self.dropout
                                               ,name='sp1'
                                               ,logging=self.logging)(plc_list[0]['feats'][1])
        
        self.feat2 = GraphConvolutionSparseK(input_dim=nfeats_in_list[2]
                                               ,output_dim=nfeats_out
                                               ,adjs=self.adjs
                                               ,features_nonzero=feats_nz_list[2]
                                               ,act=tf.nn.relu
                                               ,dropout=self.dropout
                                               ,name='sp2'
                                               ,logging=self.logging)(plc_list[0]['feats'][2])
        
        self.feat3 = GraphConvolutionSparseK(input_dim=nfeats_in_list[3]
                                               ,output_dim=nfeats_out
                                               ,adjs=self.adjs
                                               ,features_nonzero=feats_nz_list[3]
                                               ,act=tf.nn.relu
                                               ,dropout=self.dropout
                                               ,name='sp3'
                                               ,logging=self.logging)(plc_list[0]['feats'][3])
        
        self.feat4 = GraphConvolutionSparseK(input_dim=nfeats_in_list[4]
                                               ,output_dim=nfeats_out
                                               ,adjs=self.adjs
                                               ,features_nonzero=feats_nz_list[4]
                                               ,act=tf.nn.relu
                                               ,dropout=self.dropout
                                               ,name='sp4'
                                               ,logging=self.logging)(plc_list[0]['feats'][4])
        
        self.feats_one = tf.concat([self.feat0, self.feat1, self.feat2, self.feat3, self.feat4], axis=-1)
        
        plc_list[0]['features'] = self.feats_one
        self.vemb0 = GCNModelVAEK(plc_list[0], 5*nfeats_out, nnodes_list[0], nhids_list[0])
        
        self.sum_mask = tf.transpose(tf.sparse_tensor_to_dense(plc_list[0]['feats'][4], validate_indices=False))
        self.mask_tile = tf.tile(tf.expand_dims(self.sum_mask, -1), [1,1,nhids_list[0][-1]])
        self.feats_tile = tf.tile(tf.expand_dims(self.vemb0.z_mean, 0), [nnodes_list[1], 1, 1])
        self.feats_tm = tf.multiply(self.mask_tile, self.feats_tile)
        self.fc = slim.fully_connected(self.feats_tm, nhids_list[0][-1], activation_fn=tf.nn.tanh)
        self.em = tf.squeeze(tf.layers.dense(self.fc, 1, activation=None, use_bias=False))
        self.alp = tf.nn.softmax(self.em, axis=-1)
        self.alpha = tf.multiply(tf.multiply(self.sum_mask, self.alp), tf.reciprocal(tf.reduce_sum(self.alp, axis=-1, keepdims=True)))
        self.masked_inp = tf.matmul(self.alpha, self.vemb0.z_mean)
        plc_list[1]['features'] = self.masked_inp
        self.vemb1 = GCNModelVAE(plc_list[1], nhids_list[0][-1], nnodes_list[1], nhids_list[1])    



class OptimizerHierEmb(object):
    def __init__(self, plc_list, model, nnodes_list, pw_list, norm_list, learning_rate):
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        preds0 = model.vemb0.reconstructions
        labels0 = tf.reshape(tf.sparse_tensor_to_dense(plc_list[0]['adj_orig'], validate_indices=False), [-1])
        
        preds1 = model.vemb1.reconstructions
        labels1 = tf.reshape(tf.sparse_tensor_to_dense(plc_list[1]['adj_orig'], validate_indices=False), [-1])

        self.ll0 = norm_list[0] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds0
                                                                                          ,targets=labels0
                                                                                          ,pos_weight=pw_list[0]))
        
        self.ll1 = norm_list[1] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds1
                                                                                          ,targets=labels1
                                                                                          ,pos_weight=pw_list[1]))

        self.kl0 = (0.5 / nnodes_list[0]) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.vemb0.z_log_std 
                                                                         - tf.square(model.vemb0.z_mean) 
                                                                         - tf.square(tf.exp(model.vemb0.z_log_std))
                                                                         , 1))
        
        self.kl1 = (0.5 / nnodes_list[1]) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.vemb1.z_log_std 
                                                                         - tf.square(model.vemb1.z_mean) 
                                                                         - tf.square(tf.exp(model.vemb1.z_log_std))
                                                                         , 1))
        
        self.cost0 = self.ll0 - self.kl0
        self.cost1 = self.ll1 - self.kl1
        self.cost = self.cost0 + self.cost1

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        self.opt_op0 = self.optimizer.minimize(self.cost0)
        self.grads_vars0 = self.optimizer.compute_gradients(self.cost0)

        self.correct_prediction0 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds0), 0.5), tf.int32)
                                            ,tf.cast(labels0, tf.int32))
        self.accuracy0 = tf.reduce_mean(tf.cast(self.correct_prediction0, tf.float32))
        
        self.correct_prediction1 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds1), 0.5), tf.int32)
                                            ,tf.cast(labels1, tf.int32))
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))




# define placeholders

plch0 = {'feats': [tf.sparse_placeholder(tf.float32, att0.shape)
                   ,tf.sparse_placeholder(tf.float32, att1.shape)
                   ,tf.sparse_placeholder(tf.float32, att2.shape)
                   ,tf.sparse_placeholder(tf.float32, att3.shape)
                   ,tf.sparse_placeholder(tf.float32, att4.shape)]
         ,'adjs': [tf.sparse_placeholder(tf.float32, adjs_g_orig[0].shape) for _ in xrange(6)]
         ,'adj_orig': tf.sparse_placeholder(tf.float32, adjs_g_orig[0].shape)
         ,'dropout': tf.placeholder_with_default(0., shape=())}

plch1 = {'adj': tf.sparse_placeholder(tf.float32, adj_d_orig.shape)
         ,'adj_orig': tf.sparse_placeholder(tf.float32, adj_d_orig.shape)
         ,'dropout': tf.placeholder_with_default(0., shape=())}

plc_list = [plch0, plch1]



# making inputs

nfeats_in_list = [num_ndatts0, num_ndatts1, num_ndatts2, num_ndatts3, num_ndatts4]
nfeats_out = args.nfeats
nnodes_list = [num_nodes_g, num_nodes_d]
feats_nz_list = [ndatts_nonzero0, ndatts_nonzero1, ndatts_nonzero2, ndatts_nonzero3, ndatts_nonzero4]
feats_list = [ndatts0, ndatts1, ndatts2, ndatts3, ndatts4]
nhids_list = [[args.hg, args.zg], [args.hd, args.zd]]
dropout = args.dropout
learning_rate = args.lr

pos_weight0 = float(adjs_g_orig[0].shape[0] * adjs_g_orig[0].shape[0] 
                    - adjs_g_orig[0].sum()) / adjs_g_orig[0].sum()
norm0 = adjs_g_orig[0].shape[0] * adjs_g_orig[0].shape[0] / float((adjs_g_orig[0].shape[0] 
                                                                   * adjs_g_orig[0].shape[0] 
                                                                   - adjs_g_orig[0].sum()) * 2)

pos_weight1 = float(adj_d_train.shape[0] * adj_d_train.shape[0] - adj_d_train.sum()) / adj_d_train.sum()
norm1 = adj_d_train.shape[0] * adj_d_train.shape[0] / float((adj_d_train.shape[0] * adj_d_train.shape[0] 
                                                             - adj_d_train.sum()) * 2)

pw_list = [pos_weight0, pos_weight1]
norm_list = [norm0, norm1]

model = HierEmb(plc_list,nfeats_in_list,nfeats_out,nnodes_list,feats_nz_list,feats_list,nhids_list,dropout)
opt = OptimizerHierEmb(plc_list, model, nnodes_list, pw_list, norm_list, learning_rate)



#training

epochs = args.epochs

llg, klg, lld, kld = [], [], [], []

feed_dict = dict()
feed_dict.update({plch0['feats'][0]: feats_list[0]})
feed_dict.update({plch0['feats'][1]: feats_list[1]})
feed_dict.update({plch0['feats'][2]: feats_list[2]})
feed_dict.update({plch0['feats'][3]: feats_list[3]})
feed_dict.update({plch0['feats'][4]: feats_list[4]})

feed_dict.update({plch0['adjs'][0]: adjs_g_norm[0]})
feed_dict.update({plch0['adjs'][1]: adjs_g_norm[1]})
feed_dict.update({plch0['adjs'][2]: adjs_g_norm[2]})
feed_dict.update({plch0['adjs'][3]: adjs_g_norm[3]})
feed_dict.update({plch0['adjs'][4]: adjs_g_norm[4]})
feed_dict.update({plch0['adjs'][5]: adjs_g_norm[5]})

feed_dict.update({plch0['adj_orig']: adj_g_label})
feed_dict.update({plch0['dropout']: dropout})
feed_dict.update({plch1['adj']: adj_d_norm})
feed_dict.update({plch1['adj_orig']: adj_d_label})
feed_dict.update({plch1['dropout']: dropout})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    print("Node embedding using HVGAE for GG and DD graphs!")
    
    for epoch in range(epochs):
        outs = sess.run([opt.opt_op, opt.cost, opt.ll0, opt.ll1, opt.kl0, opt.kl1], feed_dict=feed_dict)
        
        outs2 = sess.run([model.vemb1.z_mean, model.vemb1.z_log_std, model.vemb0.z_mean, model.vemb0.z_log_std, model.alpha]
                         , feed_dict=feed_dict)
        
        llg.append(outs[2])
        lld.append(outs[3])
        klg.append(outs[4])
        kld.append(outs[5])
        
        roc_g, ap_g = get_roc_score(ep_g, en_g, outs2[2], adj_g_recon)
        roc_d, ap_d = get_roc_score(ep_d, en_d, outs2[0], adj_d_orig)
        
        print("Epoch:", '%03d' % (epoch + 1), "ll_g=", "{:.5f}".format(llg[-1]),
              "kl_g=", "{:.5f}".format(klg[-1]), "roc_g=", "{:.5f}".format(roc_g),
              "ap_g=", "{:.5f}".format(ap_g),"ll_d=", "{:.5f}".format(lld[-1]),
              "kl_d=", "{:.5f}".format(kld[-1]), "roc_d=", "{:.5f}".format(roc_d),
              "ap_d=", "{:.5f}".format(ap_d))
        sys.stdout.flush()
        if (epoch+1)%50==0:
            saver.save(sess, "./check_points/model_"+str(epoch)+".ckpt")
            #quit()    
    print("Optimization Finished!")
    sys.stdout.flush()


# saving results

np.savetxt('z_d_mean.csv', outs2[0], delimiter=',')
np.savetxt('z_d_log_std.csv', outs2[1], delimiter=',')
np.savetxt('z_g_mean.csv', outs2[2], delimiter=',')
np.savetxt('z_g_log_std.csv', outs2[3], delimiter=',')
np.savetxt('alpha.csv', outs2[4], delimiter=',')
