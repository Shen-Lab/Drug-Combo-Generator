from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from tensorboardX import SummaryWriter
from baselines.ppo1.gcn_policy import discriminator,discriminator_net,embnet,oracle
import os
import copy
import random
import math

from keras.layers import Input,Reshape,Embedding,GRU,LSTM,Conv1D,LeakyReLU,MaxPooling1D,concatenate,Dropout,Dense,LeakyReLU,TimeDistributed
from keras import regularizers
from keras.optimizers import SGD,Adam
from keras.losses import mean_squared_error
from keras.models import Model
import keras.backend as K
from keras.activations import relu
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint,TensorBoard
import csv
import re
def sentence_to_token(sentence, vocabulary,_WORD_SPLIT,MAX_size):
  UNK_ID = 3
  PAD_ID = 0
  words = []
  for space_separated_fragment in sentence.strip().split():
    l = _WORD_SPLIT.split(space_separated_fragment)
    words.extend(l)
  words = [w for w in words if w]
  words = [vocabulary.get(w, UNK_ID) for w in words]

  source_ids = [int(x) for x in words]
  if len(source_ids) < MAX_size:
     pad = [PAD_ID] * (MAX_size - len(source_ids))
     source_ids = list(source_ids + pad)
  elif len(source_ids) == MAX_size:
     source_ids = list(source_ids)
  else:
     source_ids = list(source_ids[:MAX_size])

  return source_ids


num_feat = 10
num_proj = 10
proj_type = "linear"
lambda1 = 20
lambda2 = 10
nepis = 10

def get_binding(seg,loss_func,disease_1hop,disease_1hop_name,disease_genes):
    disease_1hop = np.array(disease_1hop)[disease_genes]
    disease_1hop_name = np.array(disease_1hop_name)[disease_genes]
    num_prot = disease_1hop.shape[0]
    binding = np.zeros((len(seg),num_prot))
    size = 64
    binding_thr = 6
    num = math.ceil(num_prot/size)
    for i in range(len(seg)):
        print(i)
        drugs = np.tile(np.expand_dims(np.array(seg[i]),axis=0),[num_prot,1])
        for j in range(num):
           if j == num -1:
               d_temp = drugs[(num - 1)*size:num_prot,:]
               p_temp = disease_1hop[(num - 1)*size:num_prot,:]
               binding[i,(num - 1)*size:num_prot] = np.squeeze(loss_func(p_temp,d_temp),axis=-1)
           else:
               d_temp = drugs[size*j:size*(j+1),:]
               p_temp = disease_1hop[size*j:size*(j+1),:]
               binding[i,size*j:size*(j+1)] = np.squeeze(loss_func(p_temp,d_temp),axis=-1)


    #gene_chosen = disease_1hop_name[np.argsort(binding,axis=1)[:,-11:-1]].tolist()
    gene_chosen = []
    for i in range(len(seg)):
        gene_chosen.append(disease_1hop_name[np.where(binding[i,:] >= binding_thr )].tolist())
        

    binding[np.where(binding < binding_thr )] = 0
    binding[np.where(binding >= binding_thr )] = 1

    return binding, gene_chosen


def get_classifier_reward(binding1,binding2):
    reward = np.sum(np.logical_xor(binding1,binding2),axis=1)/binding1.shape[1]
    adverse = np.sum(np.logical_and(binding1,binding2),axis=1)/binding1.shape[1]
    d1 = np.sum(binding1,axis=1)/binding1.shape[1]
    d2 = np.sum(binding2,axis=1)/binding2.shape[1]
    return reward,adverse,d1,d2


def get_reward(smi1, smi2, loss_func1,loss_func2,disease_1hop,disease_1hop_name,disease_genes):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    binding1,gene_chosen1 = get_binding(smi1,loss_func1,disease_1hop,disease_1hop_name,disease_genes)
    binding2,gene_chosen2 = get_binding(smi2,loss_func2,disease_1hop,disease_1hop_name,disease_genes)
    temp_loss,adverse,binding_d1,binding_d2 = get_classifier_reward(binding1,binding2)
    return temp_loss,adverse,binding_d1,binding_d2,gene_chosen1,gene_chosen2



def deepaffinity(args, env1, env2, policy_fn, 
        vocab_comp,
        num_disease,disease_id,disease_1hop_name,
        *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        writer=None
        ):
    # Setup losses and stuff
    # ----------------------------------------
    # ----------------------------------------
    
    ob_space1 = env1.observation_space
    ac_space1 = env1.action_space
    disease_dim1 = env1.disease_feat.shape[1]
    pi1 = policy_fn("pi1", ob_space1, ac_space1, disease_dim1) # Construct network for new policy
    oldpi1 = policy_fn("oldpi1", ob_space1, ac_space1, disease_dim1) # Network for old policy
    atarg1 = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret1 = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult1 = tf.placeholder(name='lrmult1', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param1 = clip_param * lrmult1 # Annealed cliping parameter epislon
    # ----------------------------------------
    ob_space2 = env2.observation_space
    ac_space2 = env2.action_space
    disease_dim2 = env1.disease_feat.shape[1]
    pi2 = policy_fn("pi2", ob_space2, ac_space2, disease_dim2) # Construct network for new policy
    oldpi2 = policy_fn("oldpi2", ob_space2, ac_space2, disease_dim2) # Network for old policy
    atarg2 = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret2 = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    
    lrmult2 = tf.placeholder(name='lrmult2', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param2 = clip_param * lrmult2 # Annealed cliping parameter epislon
    
    # ----------------------------------------
    # ----------------------------------------
    
    # ob = U.get_placeholder_cached(name="ob")
    ob1 = {}
    ob1['adj'] = U.get_placeholder_cached(name="adj1")
    ob1['node'] = U.get_placeholder_cached(name="node1")
    

    ob_gen1 = {}
    ob_gen1['adj'] = U.get_placeholder(shape=[None, ob_space1['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen1')
    ob_gen1['node'] = U.get_placeholder(shape=[None, 1, None, ob_space1['node'].shape[2]], dtype=tf.float32,name='node_gen1')

    ob_real1 = {}
    ob_real1['adj'] = U.get_placeholder(shape=[None,ob_space1['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real1')
    ob_real1['node'] = U.get_placeholder(shape=[None,1,None,ob_space1['node'].shape[2]],dtype=tf.float32,name='node_real1')

    
    # ac = pi.pdtype.sample_placeholder([None])
    # ac = tf.placeholder(dtype=tf.int64,shape=env.action_space.nvec.shape)
    disease_dim = 16
    ac1 = tf.placeholder(dtype=tf.int64, shape=[None,4], name='ac_real1')

    disease = U.get_placeholder(shape=[None,disease_dim ], dtype=tf.float32,name='disease')
    # ----------------------------------------
    # ob = U.get_placeholder_cached(name="ob")
    ob2 = {}
    ob2['adj'] = U.get_placeholder_cached(name="adj2")
    ob2['node'] = U.get_placeholder_cached(name="node2")
    
    ob_gen2 = {}
    ob_gen2['adj'] = U.get_placeholder(shape=[None, ob_space2['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen2')
    ob_gen2['node'] = U.get_placeholder(shape=[None, 1, None, ob_space2['node'].shape[2]], dtype=tf.float32,name='node_gen2')
    
    ob_real2 = {}
    ob_real2['adj'] = U.get_placeholder(shape=[None,ob_space2['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real2')
    ob_real2['node'] = U.get_placeholder(shape=[None,1,None,ob_space2['node'].shape[2]],dtype=tf.float32,name='node_real2')
    
    # ac = pi.pdtype.sample_placeholder([None])
    # ac = tf.placeholder(dtype=tf.int64,shape=env.action_space.nvec.shape)
    ac2 = tf.placeholder(dtype=tf.int64, shape=[None,4], name='ac_real2')
    
    # ----------------------------------------
    # ----------------------------------------
   

    prot_data_class = Input(shape=(152,))
    drug1_data_class = Input(shape=(100,)) 
    drug2_data_class = Input(shape=(100,))
    linear1 = oracle(prot_data_class,drug1_data_class)
    linear2 = oracle(prot_data_class,drug2_data_class)
    class_model1 = Model(inputs=[prot_data_class,drug1_data_class],outputs=[linear1])
    class_model2 = Model(inputs=[prot_data_class,drug2_data_class],outputs=[linear2])

    loss1 = linear1
    loss2 = linear2
    #emb_graph0 = embnet(ob_gen1, args)
    #emb_graph1 = embnet(ob_gen2, args)
    #embs_sum = emb_graph0 + emb_graph1

    #logits = tf.nn.dropout(embs_sum, 0.5)
    #logits = tf.layers.dense(embs_sum, 64, activation=tf.nn.relu,name = 'class_fc1')
    #logits = tf.nn.dropout(embs_sum, 0.5)
    #logits = tf.layers.dense(logits, 64, activation=tf.nn.relu,name = 'class_fc2')
    #logits = tf.nn.dropout(embs_sum, 0.5)
    #logits = tf.layers.dense(logits, 1, activation=None,name = 'class_fc3')
    #preds = tf.nn.sigmoid(logits)

    #loss = - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits) 
    #loss = preds
    #loss = tf.zeros_like(preds)

    #loss_class_func = U.function([ob_gen1['adj'], ob_gen1['node'],ob_gen2['adj'], ob_gen2['node']],loss)
    loss_class_func1 = U.function([prot_data_class,drug1_data_class],loss1)
    loss_class_func2 = U.function([prot_data_class,drug2_data_class],loss2)
    ## PPO loss
    kloldnew1 = oldpi1.pd.kl(pi1.pd)
    ent1 = pi1.pd.entropy()
    meankl1 = tf.reduce_mean(kloldnew1)
    meanent1 = tf.reduce_mean(ent1)
    pol_entpen1 = (-entcoeff) * meanent1

    pi_logp1 = pi1.pd.logp(ac1)
    oldpi_logp1 = oldpi1.pd.logp(ac1)
    ratio_log1 = pi1.pd.logp(ac1) - oldpi1.pd.logp(ac1)

    ratio1 = tf.exp(pi1.pd.logp(ac1) - oldpi1.pd.logp(ac1)) # pnew / pold
    surr11 = ratio1 * atarg1 # surrogate from conservative policy iteration
    surr21 = tf.clip_by_value(ratio1, 1.0 - clip_param1, 1.0 + clip_param1) * atarg1 #
    pol_surr1 = - tf.reduce_mean(tf.minimum(surr11, surr21)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss1 = tf.reduce_mean(tf.square(pi1.vpred - ret1))
    total_loss1 = pol_surr1 + pol_entpen1 + vf_loss1
    losses1 = [pol_surr1, pol_entpen1, vf_loss1, meankl1, meanent1]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    # ----------------------------------------
    kloldnew2 = oldpi2.pd.kl(pi2.pd)
    ent2 = pi2.pd.entropy()
    meankl2 = tf.reduce_mean(kloldnew2)
    meanent2 = tf.reduce_mean(ent2)
    pol_entpen2 = (-entcoeff) * meanent2
    
    pi_logp2 = pi2.pd.logp(ac2)
    oldpi_logp2 = oldpi2.pd.logp(ac2)
    ratio_log2 = pi2.pd.logp(ac2) - oldpi2.pd.logp(ac2)
    
    ratio2 = tf.exp(pi2.pd.logp(ac2) - oldpi2.pd.logp(ac2)) # pnew / pold
    surr12 = ratio2 * atarg2 # surrogate from conservative policy iteration
    surr22 = tf.clip_by_value(ratio2, 1.0 - clip_param2, 1.0 + clip_param2) * atarg2 #
    pol_surr2 = - tf.reduce_mean(tf.minimum(surr12, surr22)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss2 = tf.reduce_mean(tf.square(pi2.vpred - ret2))
    total_loss2 = pol_surr2 + pol_entpen2 + vf_loss2
    losses2 = [pol_surr2, pol_entpen2, vf_loss2, meankl2, meanent2]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    
    # ----------------------------------------
    # ----------------------------------------

    ## Expert loss
    loss_expert1 = -tf.reduce_mean(pi_logp1)

    ## Discriminator loss
    # loss_d_step, _, _ = discriminator(ob_real, ob_gen,args, name='d_step')
    # loss_d_gen_step,_ = discriminator_net(ob_gen,args, name='d_step')
    # loss_d_final, _, _ = discriminator(ob_real, ob_gen,args, name='d_final')
    # loss_d_gen_final,_ = discriminator_net(ob_gen,args, name='d_final')


    if args.gan_type=='normal':
        step_pred_real1, step_logit_real1 = discriminator_net(ob_real1, num_feat,args, name='d_step')
        step_pred_gen1, step_logit_gen1 = discriminator_net(ob_gen1, num_feat,args, name='d_step')
        loss_d_step_real1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real1, labels=tf.ones_like(step_logit_real1)*0.9))
        loss_d_step_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen1, labels=tf.zeros_like(step_logit_gen1)))
        loss_d_step1 = loss_d_step_real1 + loss_d_step_gen1
        loss_g_step_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen1, labels=tf.zeros_like(step_logit_gen1)))
    elif args.gan_type=='recommend':
        step_pred_real1, step_logit_real1 = discriminator_net(ob_real1, num_feat,args, name='d_step')
        step_pred_gen1, step_logit_gen1 = discriminator_net(ob_gen1, num_feat,args, name='d_step')
        loss_d_step_real1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real1, labels=tf.ones_like(step_logit_real1)*0.9))
        loss_d_step_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen1, labels=tf.zeros_like(step_logit_gen1)))
        loss_d_step1 = loss_d_step_real1 + loss_d_step_gen1
        loss_g_step_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen1, labels=tf.ones_like(step_logit_gen1)*0.9))
    elif args.gan_type=='wgan':
        loss_d_step1, loss_g_step_gen1, _ = discriminator(ob_real1, ob_gen1, num_feat, num_proj, proj_type, lambda1, lambda2, args, name='d_step')
        #loss_d_step = loss_d_step*-1
        #loss_g_step_gen,_ = discriminator_net(ob_gen,args, name='d_step')
    # ----------------------------------------
    loss_expert2 = -tf.reduce_mean(pi_logp2)
    
    ## Discriminator loss
    # loss_d_step, _, _ = discriminator(ob_real, ob_gen,args, name='d_step')
    # loss_d_gen_step,_ = discriminator_net(ob_gen,args, name='d_step')
    # loss_d_final, _, _ = discriminator(ob_real, ob_gen,args, name='d_final')
    # loss_d_gen_final,_ = discriminator_net(ob_gen,args, name='d_final')

    if args.gan_type=='normal':
        step_pred_real2, step_logit_real2 = discriminator_net(ob_real2, num_feat, args, name='d_step')
        step_pred_gen2, step_logit_gen2 = discriminator_net(ob_gen2, num_feat, args, name='d_step')
        loss_d_step_real2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real2, labels=tf.ones_like(step_logit_real2)*0.9))
        loss_d_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.zeros_like(step_logit_gen2)))
        loss_d_step2 = loss_d_step_real2 + loss_d_step_gen2
        loss_g_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.zeros_like(step_logit_gen)))
    elif args.gan_type=='recommend':
        step_pred_real2, step_logit_real2 = discriminator_net(ob_real2, num_feat, args, name='d_step')
        step_pred_gen2, step_logit_gen2 = discriminator_net(ob_gen2, num_feat, args, name='d_step')
        loss_d_step_real2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real2, labels=tf.ones_like(step_logit_real2)*0.9))
        loss_d_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.zeros_like(step_logit_gen2)))
        loss_d_step2 = loss_d_step_real2 + loss_d_step_gen2
        loss_g_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.ones_like(step_logit_gen)*0.9))
    elif args.gan_type=='wgan':
        loss_d_step2, loss_g_step_gen2, _ = discriminator(ob_real2, ob_gen2, num_feat, num_proj, proj_type, lambda1, lambda2, args, name='d_step')
    #loss_d_step = loss_d_step*-1
    #loss_g_step_gen,_ = discriminator_net(ob_gen,args, name='d_step')

    # ----------------------------------------
    # ----------------------------------------

#    loss_d_step = loss_d_step2 + loss_d_step1
#    loss_g_step_gen = loss_g_step_gen2 + loss_g_step_gen1

    # ----------------------------------------
    # ----------------------------------------

    final_pred_real1, final_logit_real1 = discriminator_net(ob_real1, num_feat, args, name='d_final')
    final_pred_gen1, final_logit_gen1 = discriminator_net(ob_gen1, num_feat, args, name='d_final')
    loss_d_final_real1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_real1, labels=tf.ones_like(final_logit_real1)*0.9))
    loss_d_final_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen1, labels=tf.zeros_like(final_logit_gen1)))
    loss_d_final1 = loss_d_final_real1 + loss_d_final_gen1
    if args.gan_type == 'normal':
        loss_g_final_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen1, labels=tf.zeros_like(final_logit_gen1)))
    elif args.gan_type == 'recommend':
        loss_g_final_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen1, labels=tf.ones_like(final_logit_gen1)*0.9))
    elif args.gan_type=='wgan':
        loss_d_final1, loss_g_final_gen1, _ = discriminator(ob_real1, ob_gen1, num_feat, num_proj, proj_type, lambda1, lambda2, args, name='d_final')
        #loss_d_final = loss_d_final*-1
        #loss_g_final_gen,_ = discriminator_net(ob_gen,args, name='d_final')
    # ----------------------------------------
    final_pred_real2, final_logit_real2 = discriminator_net(ob_real2, num_feat, args, name='d_final')
    final_pred_gen2, final_logit_gen2 = discriminator_net(ob_gen2, num_feat, args, name='d_final')
    loss_d_final_real2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_real2, labels=tf.ones_like(final_logit_real2)*0.9))
    loss_d_final_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen2, labels=tf.zeros_like(final_logit_gen2)))
    loss_d_final2 = loss_d_final_real2 + loss_d_final_gen2
    if args.gan_type == 'normal':
        loss_g_final_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen2, labels=tf.zeros_like(final_logit_gen2)))
    elif args.gan_type == 'recommend':
        loss_g_final_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen2, labels=tf.ones_like(final_logit_gen2)*0.9))
    elif args.gan_type=='wgan':
        loss_d_final2, loss_g_final_gen2, _ = discriminator(ob_real2, ob_gen2, num_feat, num_proj, proj_type, lambda1, lambda2, args, name='d_final')
    #loss_d_final = loss_d_final*-1
    #loss_g_final_gen,_ = discriminator_net(ob_gen,args, name='d_final')

    # ----------------------------------------
    # ----------------------------------------

#    loss_d_final = loss_d_final2 + loss_d_final1
#    loss_g_final_gen = loss_g_final_gen2 + loss_g_final_gen1

    # ----------------------------------------
    # ----------------------------------------

    var_list_pi1 = pi1.get_trainable_variables()
    var_list_pi_stop1 = [var for var in var_list_pi1 if ('emb' in var.name) or ('gcn' in var.name) or ('stop' in var.name)]
    
    var_list_pi2 = pi2.get_trainable_variables()
    var_list_pi_stop2 = [var for var in var_list_pi2 if ('emb' in var.name) or ('gcn' in var.name) or ('stop' in var.name)]
    
    var_list_d_step = [var for var in tf.global_variables() if 'd_step' in var.name]
    var_list_d_final = [var for var in tf.global_variables() if 'd_final' in var.name]

    var_list_classifier =  [var for var in tf.global_variables() if 'class' in var.name]
    # ----------------------------------------
    # ----------------------------------------
    ## debug
    debug={}

    ## loss update function
    lossandgrad_ppo1 = U.function([ob1['adj'], ob1['node'], ac1, pi1.ac_real, oldpi1.ac_real, atarg1, ret1, lrmult1,disease], losses1 + [U.flatgrad(total_loss1, var_list_pi1)])
    lossandgrad_expert1 = U.function([ob1['adj'], ob1['node'], ac1, pi1.ac_real,disease], [loss_expert1, U.flatgrad(loss_expert1, var_list_pi1)])
    lossandgrad_expert_stop1 = U.function([ob1['adj'], ob1['node'], ac1, pi1.ac_real,disease], [loss_expert1, U.flatgrad(loss_expert1, var_list_pi_stop1)])
    # ----------------------------------------
    lossandgrad_ppo2 = U.function([ob2['adj'], ob2['node'], ac2, pi2.ac_real, oldpi2.ac_real, atarg2, ret2, lrmult2,disease], losses2 + [U.flatgrad(total_loss2, var_list_pi2)])
    lossandgrad_expert2 = U.function([ob2['adj'], ob2['node'], ac2, pi2.ac_real,disease], [loss_expert2, U.flatgrad(loss_expert2, var_list_pi2)])
    lossandgrad_expert_stop2 = U.function([ob2['adj'], ob2['node'], ac2, pi2.ac_real,disease], [loss_expert2, U.flatgrad(loss_expert2, var_list_pi_stop2)])

    # ----------------------------------------
    # ----------------------------------------

    lossandgrad_d_step1 = U.function([ob_real1['adj'], ob_real1['node'], ob_gen1['adj'], ob_gen1['node']], [loss_d_step1, U.flatgrad(loss_d_step1, var_list_d_step)])
    lossandgrad_d_final1 = U.function([ob_real1['adj'], ob_real1['node'], ob_gen1['adj'], ob_gen1['node']], [loss_d_final1, U.flatgrad(loss_d_final1, var_list_d_final)])

    loss_g_gen_step_func1 = U.function([ob_gen1['adj'], ob_gen1['node']], loss_g_step_gen1)
    loss_g_gen_final_func1 = U.function([ob_gen1['adj'], ob_gen1['node']], loss_g_final_gen1)
    # ----------------------------------------
    lossandgrad_d_step2 = U.function([ob_real2['adj'], ob_real2['node'], ob_gen2['adj'], ob_gen2['node']], [loss_d_step2, U.flatgrad(loss_d_step2, var_list_d_step)])
    lossandgrad_d_final2 = U.function([ob_real2['adj'], ob_real2['node'], ob_gen2['adj'], ob_gen2['node']], [loss_d_final2, U.flatgrad(loss_d_final2, var_list_d_final)])

    loss_g_gen_step_func2 = U.function([ob_gen2['adj'], ob_gen2['node']], loss_g_step_gen2)
    loss_g_gen_final_func2 = U.function([ob_gen2['adj'], ob_gen2['node']], loss_g_final_gen2)

    # ----------------------------------------
    # ----------------------------------------

    adam_pi1 = MpiAdam(var_list_pi1, epsilon=adam_epsilon)
    adam_pi_stop1 = MpiAdam(var_list_pi_stop1, epsilon=adam_epsilon)

    adam_pi2 = MpiAdam(var_list_pi2, epsilon=adam_epsilon)
    adam_pi_stop2 = MpiAdam(var_list_pi_stop2, epsilon=adam_epsilon)

    # ----------------------------------------
    # ----------------------------------------

    adam_d_step = MpiAdam(var_list_d_step, epsilon=adam_epsilon)
    adam_d_final = MpiAdam(var_list_d_final, epsilon=adam_epsilon)

    # ----------------------------------------
    # ----------------------------------------

    assign_old_eq_new1 = U.function([],[], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi1.get_variables(), pi1.get_variables())])
    #
    # compute_losses_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real],
    #                                 loss_expert)
    compute_losses1 = U.function([ob1['adj'], ob1['node'], ac1, pi1.ac_real, oldpi1.ac_real, atarg1, ret1, lrmult1,disease], losses1)
    # ----------------------------------------
    assign_old_eq_new2 = U.function([],[], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi2.get_variables(), pi2.get_variables())])
    #
    # compute_losses_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real],
    #                                 loss_expert)
    compute_losses2 = U.function([ob2['adj'], ob2['node'], ac2, pi2.ac_real, oldpi2.ac_real, atarg2, ret2, lrmult2,disease], losses2)

    # ----------------------------------------
    # ----------------------------------------

    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # ----------------------------------------
    # ----------------------------------------

    lenbuffer1 = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid1 = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer1 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env1 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_step1 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_final1 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final1 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final_stat1 = deque(maxlen=100) # rolling buffer for episode rewardsn
    # ----------------------------------------
    lenbuffer2 = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid2 = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer2 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env2 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_step2 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_final2 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final2 = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final_stat2 = deque(maxlen=100) # rolling buffer for episode rewardsn

    # ----------------------------------------
    # ----------------------------------------

    classifier_buffer = deque(maxlen=1)   
    classifier_adverse_buffer = deque(maxlen=1) 
    classifier_binding1_buffer = deque(maxlen=1)
    classifier_binding2_buffer = deque(maxlen=1)

    # ----------------------------------------
    # ----------------------------------------

    #disease_count = 0
    #disease_list = list(range(num_disease))
    #random.shuffle(disease_list)

    class_model1.load_weights('ckpt/weights.best.hdf5')
    class_model2.load_weights('ckpt/weights.best.hdf5')
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    if args.load==1:
        try:
            fname1 = './ckpt/' + args.name_full_load1
            fname2 = './ckpt/' + args.name_full_load2
            sess = tf.get_default_session()
            saver1 = tf.train.Saver(var_list_pi1)
            saver2 = tf.train.Saver(var_list_pi2)
            saver1.restore(sess, fname1)
            saver2.restore(sess, fname2)
            iters_so_far = int(fname1.split('_')[-2])+1
            print('model restored!', fname1, 'iters_so_far:', iters_so_far,flush=True)
            print('model restored!', fname2, 'iters_so_far:', iters_so_far,flush=True)
        except:
            print(fname,'ckpt not found, start with iters 0')

    adam_pi1.sync()
    adam_pi_stop1.sync()

    adam_pi2.sync()
    adam_pi_stop2.sync()

    adam_d_step.sync()
    adam_d_final.sync()

    cwd = os.path.dirname(__file__)

    drug_max_size = 100
    WORD_SPLIT_comp = re.compile(b"(\S)")
    drug_gen1_path = os.path.join(os.path.dirname(cwd),'../../drug_eval1.txt')


    drug_gen2_path = os.path.join(os.path.dirname(cwd),'../../drug_eval2.txt')
    ob,disease_feat,disease_1hop,disease_genes = env1.reset(disease_id)

    smi_set1 = []
    with open(drug_gen1_path) as f:
         for line in f:
             line = line.strip()
             smi_set1.append(line)

    smi_set2 = []
    with open(drug_gen2_path) as f:
         for line in f:
             line = line.strip()
             smi_set2.append(line)

    smi_code1 = [] 
    smi_code2 = []
    for i in range(len(smi_set2)):
        smi_code1.append(sentence_to_token(tf.compat.as_bytes(smi_set1[i]),vocab_comp,WORD_SPLIT_comp,drug_max_size))
        smi_code2.append(sentence_to_token(tf.compat.as_bytes(smi_set2[i]),vocab_comp,WORD_SPLIT_comp,drug_max_size))
  
    reward,adverse,d1,d2,gene_chosen1,gene_chosen2 = get_reward(smi_code1, smi_code2, loss_class_func1,loss_class_func2,disease_1hop,disease_1hop_name,disease_genes)
    np.savetxt('reward_eval.txt',reward)
    np.savetxt('adverse_eval.txt',adverse)
    np.savetxt('d1_eval.txt',d1)
    np.savetxt('d2_eval.txt',d2)
    with open('gene_chosn1_eval.csv','a') as f:
         for l in gene_chosen1:
             mystr = " ".join([str(elem) for elem in l])+"\n"
             f.write(mystr)

    with open('gene_chosn2_eval.csv','a') as f:
         for l in gene_chosen2:
             mystr = " ".join([str(elem) for elem in l])+"\n"
             f.write(mystr)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
