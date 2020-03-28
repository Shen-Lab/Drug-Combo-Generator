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


num_feat = 10
num_proj = 10
proj_type = "linear"
lambda1 = 20
lambda2 = 10
nepis = 10

def traj_segment_generator(args, pi, env,disease_id, horizon, stochastic, d_step_func, d_final_func, num_episodes,env_ind):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    disease_id = yield
    ob,disease_feat,disease_1hop,disease_genes = env.reset(disease_id)
    ob_adj = ob['adj']
    ob_node = ob['node']

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_env = 0
    cur_ep_ret_d_step = 0
    cur_ep_ret_d_final = 0
    cur_ep_len = 0 # len of current episode
    cur_ep_len_valid = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_d_step = []
    ep_rets_d_final = []
    ep_rets_env = []
    ep_lens = [] # lengths of ...
    ep_lens_valid = [] # lengths of ...
    ep_rew_final = []
    ep_rew_final_stat = []
    
    cur_num_epi = 1
    #i = 0


    # Initialize history arrays
    # obs = np.array([ob for _ in range(horizon)])
    
    #ob_adjs = np.array([ob_adj for _ in range(horizon)])
    #ob_nodes = np.array([ob_node for _ in range(horizon)])
    #ob_adjs_final = []
    #ob_nodes_final = []
    #rews = np.zeros(horizon, 'float32')
    #vpreds = np.zeros(horizon, 'float32')
    #news = np.zeros(horizon, 'int32')
    #acs = np.array([ac for _ in range(horizon)])
    #prevacs = acs.copy()

    ob_adjs = []
    ob_nodes = []
    ob_adjs_final = []
    ob_smi_final = []
    ob_nodes_final = []
    rews = []
    vpreds = []
    news = []
    acs = []
    prevacs = []


    while True:

        prevac = ac
        ac, vpred, debug = pi.act(stochastic, ob,disease_feat)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if t > 0 and cur_num_epi % num_episodes == num_episodes -1:
            #i = 0
            
            ob_adjs = np.array(ob_adjs)
            ob_nodes = np.array(ob_nodes)
            rews = np.array(rews, dtype=np.float32)
            vpreds = np.squeeze(np.array(vpreds, dtype=np.float32))
            news = np.array(news, dtype=np.int32)
            acs = np.squeeze(np.array(acs))
            prevacs = np.squeeze(np.array(prevacs))
            yield {"ob_adj" : ob_adjs, "ob_node" : ob_nodes,"ob_adj_final" : np.array(ob_adjs_final), "ob_node_final" : np.array(ob_nodes_final), "rew" : rews, "vpred" : vpreds, "new" : news,"smi":np.array(ob_smi_final),"disease_1hop":disease_1hop,"disease_genes":disease_genes,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_lens_valid" : ep_lens_valid, "ep_final_rew":ep_rew_final, "ep_final_rew_stat":ep_rew_final_stat,"ep_rets_env" : ep_rets_env,"ep_rets_d_step" : ep_rets_d_step,"ep_rets_d_final" : ep_rets_d_final}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            disease_id = yield
 
            ep_rets = []
            ep_lens = []
            ep_lens_valid = []
            ep_rew_final = []
            ep_rew_final_stat = []
            ep_rets_d_step = []
            ep_rets_d_final = []
            ep_rets_env = []
            ob_adjs_final = []
            ob_smi_final = []
            ob_nodes_final = []
            ob_adjs = []
            ob_nodes = []
            rews = []
            vpreds = []
            news = []
            acs = []
            prevacs = []
            cur_num_epi = 1


        #i = t % horizon
        # obs[i] = ob
        ob_adjs.append(ob['adj'])
        ob_nodes.append(ob['node'])
        vpreds.append(vpred)
        news.append(new)
        acs.append(ac)
        prevacs.append(prevac)
        ob, rew_env, new, info,disease_feat,disease_1hop,disease_genes = env.step(ac)
        rew_d_step = 0 # default
        if rew_env>0: # if action valid
            cur_ep_len_valid += 1
            # add stepwise discriminator reward
            if args.has_d_step==1:
                if args.gan_type=='normal' or args.gan_type=='wgan':
                    rew_d_step = args.gan_step_ratio * (
                        d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])) / env.max_atom
                elif args.gan_type == 'recommend':
                    rew_d_step = args.gan_step_ratio * (
                        max(1-d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),-2)) / env.max_atom
        rew_d_final = 0 # default
        if new:
            if args.has_d_final==1:
                if args.gan_type == 'normal' or args.gan_type=='wgan':
                    rew_d_final = args.gan_final_ratio * (
                        d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]))
                elif args.gan_type == 'recommend':
                    rew_d_final = args.gan_final_ratio * (
                        max(1 - d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),
                            -2))
        #print("reward d step: "+str(rew_d_step))
        #print("reward d final: "+str(rew_d_final))
        #print(rew_d_step,rew_env,rew_d_final)
        rews.append(rew_d_step + rew_env + rew_d_final)

        cur_ep_ret += rews[-1]
        cur_ep_ret_d_step += rew_d_step
        cur_ep_ret_d_final += rew_d_final
        cur_ep_ret_env += rew_env
        cur_ep_len += 1

        if new:
            if args.env=='molecule':
                with open('molecule_gen/'+args.name_full+'_'+env_ind+'.csv', 'a') as f:
                    str = ''.join(['{},']*(len(info)+4))[:-1]+'\n'
                    f.write(str.format(info['smile'],info['smile_code'], disease_id,info['reward_valid'], info['reward_qed'], info['reward_sa'], info['final_stat'], rew_env, rew_d_step, rew_d_final, cur_ep_ret, info['flag_steric_strain_filter'], info['flag_zinc_molecule_filter'], info['stop']))
            ob_adjs_final.append(ob['adj'])
            ob_smi_final.append(info['smile_code'])
            ob_nodes_final.append(ob['node'])
            ep_rets.append(cur_ep_ret)
            ep_rets_env.append(cur_ep_ret_env)
            ep_rets_d_step.append(cur_ep_ret_d_step)
            ep_rets_d_final.append(cur_ep_ret_d_final)
            ep_lens.append(cur_ep_len)
            ep_lens_valid.append(cur_ep_len_valid)
            ep_rew_final.append(rew_env)
            ep_rew_final_stat.append(info['final_stat'])
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_len_valid = 0
            cur_ep_ret_d_step = 0
            cur_ep_ret_d_final = 0
            cur_ep_ret_env = 0
            ob,disease_feat,disease_1hop,disease_genes = env.reset(disease_id)
            cur_num_epi += 1

        t += 1
        #i += 1

def traj_final_generator(pi, env, disease_id,batch_size, stochastic):
    ob,disease_feat,disease_1hop,disease_genes = env.reset(disease_id)
    ob_adj = ob['adj']
    ob_node = ob['node']
    ob_adjs = np.array([ob_adj for _ in range(batch_size)])
    ob_nodes = np.array([ob_node for _ in range(batch_size)])
    for i in range(batch_size):
        ob,disease_feat,disease_1hop,disease_genes = env.reset(disease_id)
        while True:
            ac, vpred, debug = pi.act(stochastic, ob,disease_feat)
            ob, rew_env, new, info,disease_feat,disease_1hop,disease_genes = env.step(ac)
            np.set_printoptions(precision=2, linewidth=200)
            # print('ac',ac)
            # print('ob',ob['adj'],ob['node'])
            if new:
                ob_adjs[i]=ob['adj']
                ob_nodes[i]=ob['node']
                break
    return ob_adjs,ob_nodes


def get_binding(args,seg,loss_func,disease_genes):
    disease_1hop = np.array(seg["disease_1hop"])[disease_genes]
    num_prot = disease_1hop.shape[0]
    binding = np.zeros((len(seg["smi"]),num_prot))
    size = 64
    binding_thr = args.deepaffinity_thr
    num = math.ceil(num_prot/size)
    for i in range(len(seg["smi"])):
        print(i)
        drugs = np.tile(np.expand_dims(np.array(seg["smi"][i]),axis=0),[num_prot,1])
        for j in range(num):
           if j == num -1:
               d_temp = drugs[(num - 1)*size:num_prot,:]
               p_temp = disease_1hop[(num - 1)*size:num_prot,:]
               binding[i,(num - 1)*size:num_prot] = np.squeeze(loss_func(p_temp,d_temp),axis=-1)
           else:
               d_temp = drugs[size*j:size*(j+1),:]
               p_temp = disease_1hop[size*j:size*(j+1),:]
               binding[i,size*j:size*(j+1)] = np.squeeze(loss_func(p_temp,d_temp),axis=-1)

    binding[np.where(binding < binding_thr )] = 0
    binding[np.where(binding >= binding_thr )] = 1
    return binding


def get_classifier_reward(binding1,binding2):
    reward = np.sum(np.logical_xor(binding1,binding2),axis=1)/binding1.shape[1]
    adverse = np.sum(np.logical_and(binding1,binding2),axis=1)/binding1.shape[1]
    d1 = np.sum(binding1,axis=1)/binding1.shape[1]
    d2 = np.sum(binding2,axis=1)/binding2.shape[1]
    return reward,adverse,d1,d2


def add_vtarg_and_adv(args,seg, seg2, gamma, lam, loss_func1,loss_func2):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    binding1 = get_binding(args,seg,loss_func1,seg["disease_genes"])
    binding2 = get_binding(args,seg2,loss_func2,seg2["disease_genes"])
    temp_loss,adverse,binding_d1,binding_d2 = get_classifier_reward(binding1,binding2)
    print("cls loss:")
    print(temp_loss)
    cls_weight = args.network_weight    
    cls_loss = np.zeros_like(seg["rew"])
    T = len(seg["rew"])
    idx_new = 0
    for i in range(T):
        if seg["new"][i]:
            cls_loss[i] = temp_loss[idx_new]
            idx_new +=1
    
    seg["cls_loss"] = temp_loss
    seg["adverse"] = adverse
    seg["binding"] = binding_d1
    seg["rew"] = cls_weight *cls_loss + seg["rew"] * args.others_weight
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
   
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

    cls_loss2 = np.zeros_like(seg2["rew"])
    T2 = len(seg2["rew"])
    idx_new2 = 0
    for i in range(T2):
        if seg2["new"][i]:
            cls_loss2[i] = temp_loss[idx_new2]
            idx_new2 +=1

    seg2["cls_loss"] = temp_loss
    seg2["adverse"] = adverse
    seg2["binding"] = binding_d2
    seg2["rew"] = cls_weight * cls_loss2 + seg2["rew"] * args.others_weight
    new2 = np.append(seg2["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred2 = np.append(seg2["vpred"], seg2["nextvpred"])
    seg2["adv"] = gaelam2 = np.empty(T2, 'float32')
    rew2 = seg2["rew"]
    lastgaelam2 = 0
    for t in reversed(range(T2)):
        nonterminal2 = 1-new2[t+1]
        delta2 = rew2[t] + gamma * vpred2[t+1] * nonterminal2 - vpred2[t]
        gaelam2[t] = lastgaelam2 = delta2 + gamma * lam * nonterminal2 * lastgaelam2
    seg2["tdlamret"] = seg2["adv"] + seg2["vpred"]




def learn(args, env1, env2, policy_fn, 
        num_disease,disease_id,
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
    
    ob1 = {}
    ob1['adj'] = U.get_placeholder_cached(name="adj1")
    ob1['node'] = U.get_placeholder_cached(name="node1")
    

    ob_gen1 = {}
    ob_gen1['adj'] = U.get_placeholder(shape=[None, ob_space1['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen1')
    ob_gen1['node'] = U.get_placeholder(shape=[None, 1, None, ob_space1['node'].shape[2]], dtype=tf.float32,name='node_gen1')

    ob_real1 = {}
    ob_real1['adj'] = U.get_placeholder(shape=[None,ob_space1['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real1')
    ob_real1['node'] = U.get_placeholder(shape=[None,1,None,ob_space1['node'].shape[2]],dtype=tf.float32,name='node_real1')

    
    disease_dim = 16
    ac1 = tf.placeholder(dtype=tf.int64, shape=[None,4], name='ac_real1')

    disease = U.get_placeholder(shape=[None,disease_dim ], dtype=tf.float32,name='disease')
    ob2 = {}
    ob2['adj'] = U.get_placeholder_cached(name="adj2")
    ob2['node'] = U.get_placeholder_cached(name="node2")
    
    ob_gen2 = {}
    ob_gen2['adj'] = U.get_placeholder(shape=[None, ob_space2['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen2')
    ob_gen2['node'] = U.get_placeholder(shape=[None, 1, None, ob_space2['node'].shape[2]], dtype=tf.float32,name='node_gen2')
    
    ob_real2 = {}
    ob_real2['adj'] = U.get_placeholder(shape=[None,ob_space2['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real2')
    ob_real2['node'] = U.get_placeholder(shape=[None,1,None,ob_space2['node'].shape[2]],dtype=tf.float32,name='node_real2')
    
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
        loss_g_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.zeros_like(step_logit_gen2)))
    elif args.gan_type=='recommend':
        step_pred_real2, step_logit_real2 = discriminator_net(ob_real2, num_feat, args, name='d_step')
        step_pred_gen2, step_logit_gen2 = discriminator_net(ob_gen2, num_feat, args, name='d_step')
        loss_d_step_real2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real2, labels=tf.ones_like(step_logit_real2)*0.9))
        loss_d_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.zeros_like(step_logit_gen2)))
        loss_d_step2 = loss_d_step_real2 + loss_d_step_gen2
        loss_g_step_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen2, labels=tf.ones_like(step_logit_gen2)*0.9))
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

    seg_gen1 = traj_segment_generator(args, pi1, env1,disease_id, timesteps_per_actorbatch, True, loss_g_gen_step_func1, loss_g_gen_final_func1, nepis,'1')
    seg_gen2 = traj_segment_generator(args, pi2, env2,disease_id, timesteps_per_actorbatch, True, loss_g_gen_step_func2, loss_g_gen_final_func2, nepis,'2')


    U.initialize()
    #saver_classifier = tf.train.Saver(var_list_classifier)
    #saver_classifier.restore(tf.get_default_session(), "/scratch/user/mostafa_karimi/rlproj/checkpoint_classifier/classifier_iter_99450")
    class_model1.load_weights('./ckpt/weights.best.hdf5')
    class_model2.load_weights('./ckpt/weights.best.hdf5')
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    if args.load==1:
        try:
            fname1 = './ckpt/' + args.name_full_load1
            fname2 = './ckpt/' + args.name_full_load2
            sess = tf.get_default_session()
            # sess.run(tf.global_variables_initializer())
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

    counter = 0
    level = 0
    ## start training
    while True:
        #if disease_count == len(disease_list):
        #   disease_count = 0
        #   random.shuffle(disease_list)
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # logger.log("********** Iteration %i ************"%iters_so_far)
        seg_gen1.__next__()
        seg1 = seg_gen1.send(disease_id)
        seg_gen2.__next__()
        seg2 = seg_gen2.send(disease_id)

        add_vtarg_and_adv(args,seg1, seg2, gamma, lam, loss_class_func1,loss_class_func2)
        print("iter: ",iters_so_far,flush=True)
        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob_adj1, ob_node1, ac1, atarg1, tdlamret1 = seg1["ob_adj"], seg1["ob_node"], seg1["ac"], seg1["adv"], seg1["tdlamret"]
        vpredbefore1 = seg1["vpred"]  # predicted value function before udpate
        atarg1 = (atarg1 - atarg1.mean()) / atarg1.std()  # standardized advantage function estimate
        d1 = Dataset(dict(ob_adj=ob_adj1, ob_node=ob_node1, ac=ac1, atarg=atarg1, vtarg=tdlamret1), shuffle=not pi1.recurrent)
        optim_batchsize1 = optim_batchsize or ob_adj1.shape[0]

        ob_adj2, ob_node2, ac2, atarg2, tdlamret2 = seg2["ob_adj"], seg2["ob_node"], seg2["ac"], seg2["adv"], seg2["tdlamret"]
        vpredbefore2 = seg2["vpred"]  # predicted value function before udpate
        atarg2 = (atarg2 - atarg2.mean()) / atarg2.std()  # standardized advantage function estimate
        d2 = Dataset(dict(ob_adj=ob_adj2, ob_node=ob_node2, ac=ac2, atarg=atarg2, vtarg=tdlamret2), shuffle=not pi2.recurrent)
        optim_batchsize2 = optim_batchsize or ob_adj2.shape[0]

        
        # inner training loop, train policy
        for i_optim in range(optim_epochs):
            loss_expert1=0
            loss_expert_stop1=0
            g_expert1=0
            g_expert_stop1=0
            
            loss_expert2=0
            loss_expert_stop2=0
            g_expert2=0
            g_expert_stop2=0


            loss_d_step1 = 0
            loss_d_final1 = 0
            g_ppo1 = 0
            g_d_step1 = 0
            g_d_final1 = 0

            loss_d_step2 = 0
            loss_d_final2 = 0
            g_ppo2 = 0
            g_d_step2 = 0
            g_d_final2 = 0

            _, _,disease_feat = env1.get_expert(optim_batchsize1,disease_id)
            pretrain_shift = 5
            ## Expert
            if iters_so_far>=args.expert_start and iters_so_far<=args.expert_end+pretrain_shift:
                ob_expert1, ac_expert1,_ = env1.get_expert(optim_batchsize1,disease_id)
                loss_expert1, g_expert1 = lossandgrad_expert1(ob_expert1['adj'], ob_expert1['node'], ac_expert1, ac_expert1,disease_feat)
                loss_expert1 = np.mean(loss_expert1)
            
                ob_expert2, ac_expert2,_ = env2.get_expert(optim_batchsize2,disease_id)
                loss_expert2, g_expert2 = lossandgrad_expert2(ob_expert2['adj'], ob_expert2['node'], ac_expert2, ac_expert2,disease_feat)
                loss_expert2 = np.mean(loss_expert2)


            ## PPO
            if iters_so_far>=args.rl_start and iters_so_far<=args.rl_end:
                assign_old_eq_new1() # set old parameter values to new parameter values
                batch1 = d1.next_batch(optim_batchsize1)
                
                assign_old_eq_new2() # set old parameter values to new parameter values
                batch2 = d2.next_batch(optim_batchsize2)
               
                # ppo
                # if args.has_ppo==1:
                if iters_so_far >= args.rl_start+pretrain_shift: # start generator after discriminator trained a well..
                    *newlosses1, g_ppo1 = lossandgrad_ppo1(batch1["ob_adj"], batch1["ob_node"], batch1["ac"], batch1["ac"], batch1["ac"], batch1["atarg"], batch1["vtarg"], cur_lrmult,disease_feat)
                    losses_ppo1=newlosses1
                
                    *newlosses2, g_ppo2 = lossandgrad_ppo2(batch2["ob_adj"], batch2["ob_node"], batch2["ac"], batch2["ac"], batch2["ac"], batch2["atarg"], batch2["vtarg"], cur_lrmult,disease_feat)
                    losses_ppo2=newlosses2

                if args.has_d_step==1 and i_optim>=optim_epochs//2:
                    # update step discriminator
                    ob_expert1, _,_ = env1.get_expert(optim_batchsize1,disease_id,curriculum=args.curriculum,level_total=args.curriculum_num,level=level)
                    loss_d_step1, g_d_step1 = lossandgrad_d_step1(ob_expert1["adj"], ob_expert1["node"], batch1["ob_adj"], batch1["ob_node"])
                    adam_d_step.update(g_d_step1, optim_stepsize * cur_lrmult)
                    loss_d_step1 = np.mean(loss_d_step1)
                
                    ob_expert2, _,_ = env2.get_expert(optim_batchsize2,disease_id,curriculum=args.curriculum,level_total=args.curriculum_num,level=level)
                    loss_d_step2, g_d_step2 = lossandgrad_d_step2(ob_expert2["adj"], ob_expert2["node"], batch2["ob_adj"], batch2["ob_node"])
                    adam_d_step.update(g_d_step2, optim_stepsize * cur_lrmult)
                    loss_d_step2 = np.mean(loss_d_step2)

                if args.has_d_final==1 and i_optim>=optim_epochs//4*3:
                    # update final discriminator
                    ob_expert1, _ ,_ = env1.get_expert(optim_batchsize1, disease_id,is_final=True,
                                                  curriculum=args.curriculum,level_total=args.curriculum_num, level=level)
                    seg_final_adj1, seg_final_node1 = traj_final_generator(pi1, copy.deepcopy(env1),disease_id, optim_batchsize1, True)
                    # update final discriminator
                    loss_d_final1, g_d_final1 = lossandgrad_d_final1(ob_expert1["adj"], ob_expert1["node"], seg_final_adj1, seg_final_node1)
                    # loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], ob_adjs, ob_nodes)
                    adam_d_final.update(g_d_final1, optim_stepsize * cur_lrmult)
                    # logger.log(fmt_row(13, np.mean(losses, axis=0)))

                    ob_expert2, _,_ = env2.get_expert(optim_batchsize2, disease_id,is_final=True,
                                                  curriculum=args.curriculum,level_total=args.curriculum_num, level=level)
                    seg_final_adj2, seg_final_node2 = traj_final_generator(pi2, copy.deepcopy(env2),disease_id, optim_batchsize2, True)
                    # update final discriminator
                    loss_d_final2, g_d_final2 = lossandgrad_d_final2(ob_expert2["adj"], ob_expert2["node"], seg_final_adj2, seg_final_node2)
                    # loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], ob_adjs, ob_nodes)
                    adam_d_final.update(g_d_final2, optim_stepsize * cur_lrmult)

     
            #print("gradient1 PPO: "+str(0.2*g_ppo1)+ " Expert: "+str(0.05*g_expert1))
            #print("gradient2 PPO: "+str(0.2*g_ppo2)+ " Expert: "+str(0.05*g_expert2))
            #print("step size: "+ str(optim_stepsize)+" and "+str(cur_lrmult))
            adam_pi1.update(0.2*g_ppo1+0.05*g_expert1, optim_stepsize * cur_lrmult)
            adam_pi2.update(0.2*g_ppo2+0.05*g_expert2, optim_stepsize * cur_lrmult)

        losses1 = []
        for batch1 in d1.iterate_once(optim_batchsize1):
            newlosses1 = compute_losses1(batch1["ob_adj"], batch1["ob_node"], batch1["ac"], batch1["ac"], batch1["ac"], batch1["atarg"], batch1["vtarg"], cur_lrmult,disease_feat)
            losses1.append(newlosses1)
        meanlosses1,_,_ = mpi_moments(losses1, axis=0)

        losses2 = []
        for batch2 in d2.iterate_once(optim_batchsize2):
            newlosses2 = compute_losses2(batch2["ob_adj"], batch2["ob_node"], batch2["ac"], batch2["ac"], batch2["ac"], batch2["atarg"], batch2["vtarg"], cur_lrmult,disease_feat)
            losses2.append(newlosses2)
        meanlosses2,_,_ = mpi_moments(losses2, axis=0)

        if writer is not None:
            writer.add_scalar("loss_expert1", loss_expert1, iters_so_far)
            writer.add_scalar("loss_expert_stop1", loss_expert_stop1, iters_so_far)
            writer.add_scalar("loss_d_step1", loss_d_step1, iters_so_far)
            writer.add_scalar("loss_d_final1", loss_d_final1, iters_so_far)
            writer.add_scalar('grad_expert_min1', np.amin(g_expert1), iters_so_far)
            writer.add_scalar('grad_expert_max1', np.amax(g_expert1), iters_so_far)
            writer.add_scalar('grad_expert_norm1', np.linalg.norm(g_expert1), iters_so_far)
            writer.add_scalar('grad_expert_stop_min1', np.amin(g_expert_stop1), iters_so_far)
            writer.add_scalar('grad_expert_stop_max1', np.amax(g_expert_stop1), iters_so_far)
            writer.add_scalar('grad_expert_stop_norm1', np.linalg.norm(g_expert_stop1), iters_so_far)
            writer.add_scalar('grad_rl_min1', np.amin(g_ppo1), iters_so_far)
            writer.add_scalar('grad_rl_max1', np.amax(g_ppo1), iters_so_far)
            writer.add_scalar('grad_rl_norm1', np.linalg.norm(g_ppo1), iters_so_far)
            writer.add_scalar('g_d_step_min1', np.amin(g_d_step1), iters_so_far)
            writer.add_scalar('g_d_step_max1', np.amax(g_d_step1), iters_so_far)
            writer.add_scalar('g_d_step_norm1', np.linalg.norm(g_d_step1), iters_so_far)
            writer.add_scalar('g_d_final_min1', np.amin(g_d_final1), iters_so_far)
            writer.add_scalar('g_d_final_max1', np.amax(g_d_final1), iters_so_far)
            writer.add_scalar('g_d_final_norm1', np.linalg.norm(g_d_final1), iters_so_far)
            writer.add_scalar('learning_rate1', optim_stepsize * cur_lrmult, iters_so_far)
            

            writer.add_scalar("loss_expert2", loss_expert2, iters_so_far)
            writer.add_scalar("loss_expert_stop2", loss_expert_stop2, iters_so_far)
            writer.add_scalar("loss_d_step2", loss_d_step2, iters_so_far)
            writer.add_scalar("loss_d_final2", loss_d_final2, iters_so_far)
            writer.add_scalar('grad_expert_min2', np.amin(g_expert2), iters_so_far)
            writer.add_scalar('grad_expert_max2', np.amax(g_expert2), iters_so_far)
            writer.add_scalar('grad_expert_norm2', np.linalg.norm(g_expert2), iters_so_far)
            writer.add_scalar('grad_expert_stop_min2', np.amin(g_expert_stop2), iters_so_far)
            writer.add_scalar('grad_expert_stop_max2', np.amax(g_expert_stop2), iters_so_far)
            writer.add_scalar('grad_expert_stop_norm2', np.linalg.norm(g_expert_stop2), iters_so_far)
            writer.add_scalar('grad_rl_min2', np.amin(g_ppo2), iters_so_far)
            writer.add_scalar('grad_rl_max2', np.amax(g_ppo2), iters_so_far)
            writer.add_scalar('grad_rl_norm2', np.linalg.norm(g_ppo2), iters_so_far)
            writer.add_scalar('g_d_step_min2', np.amin(g_d_step2), iters_so_far)
            writer.add_scalar('g_d_step_max2', np.amax(g_d_step2), iters_so_far)
            writer.add_scalar('g_d_step_norm2', np.linalg.norm(g_d_step2), iters_so_far)
            writer.add_scalar('g_d_final_min2', np.amin(g_d_final2), iters_so_far)
            writer.add_scalar('g_d_final_max2', np.amax(g_d_final2), iters_so_far)
            writer.add_scalar('g_d_final_norm2', np.linalg.norm(g_d_final2), iters_so_far)
            writer.add_scalar('learning_rate2', optim_stepsize * cur_lrmult, iters_so_far)
            writer.add_scalar('rew2', optim_stepsize * cur_lrmult, iters_so_far)
            

        for (lossval, name) in zipsame(meanlosses1, loss_names):
            # logger.record_tabular("loss_"+name, lossval)
            if writer is not None:
                writer.add_scalar("loss_"+name+"_1", lossval, iters_so_far)

        for (lossval, name) in zipsame(meanlosses2, loss_names):
            # logger.record_tabular("loss_"+name, lossval)
            if writer is not None:
                writer.add_scalar("loss_"+name+"_2", lossval, iters_so_far)

        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        if writer is not None:
            writer.add_scalar("ev_tdlam_before_1", explained_variance(vpredbefore1, tdlamret1), iters_so_far)
            writer.add_scalar("ev_tdlam_before_2", explained_variance(vpredbefore2, tdlamret2), iters_so_far)
        lrlocal1 = (seg1["ep_lens"],seg1["ep_lens_valid"], seg1["ep_rets"],seg1["ep_rets_env"],seg1["ep_rets_d_step"],seg1["ep_rets_d_final"],seg1["ep_final_rew"],seg1["ep_final_rew_stat"],seg1["cls_loss"],seg1["adverse"],seg1["binding"]) # local values
        listoflrpairs1 = MPI.COMM_WORLD.allgather(lrlocal1) # list of tuples
        lens1, lens_valid1, rews1, rews_env1, rews_d_step1,rews_d_final1, rews_final1,rews_final_stat1,cls_rew,cls_adverse,binding_d1 = map(flatten_lists, zip(*listoflrpairs1))
        lenbuffer1.extend(lens1)
        lenbuffer_valid1.extend(lens_valid1)
        rewbuffer1.extend(rews1)
        rewbuffer_d_step1.extend(rews_d_step1)
        rewbuffer_d_final1.extend(rews_d_final1)
        rewbuffer_env1.extend(rews_env1)
        rewbuffer_final1.extend(rews_final1)
        rewbuffer_final_stat1.extend(rews_final_stat1)
        classifier_buffer.extend(cls_rew)
        classifier_adverse_buffer.extend(cls_adverse)
        classifier_binding1_buffer.extend(binding_d1)
        

        lrlocal2 = (seg2["ep_lens"],seg2["ep_lens_valid"], seg2["ep_rets"],seg2["ep_rets_env"],seg2["ep_rets_d_step"],seg2["ep_rets_d_final"],seg2["ep_final_rew"],seg2["ep_final_rew_stat"],seg2["binding"]) # local values
        listoflrpairs2 = MPI.COMM_WORLD.allgather(lrlocal2) # list of tuples
        lens2, lens_valid2, rews2, rews_env2, rews_d_step2,rews_d_final2, rews_final2,rews_final_stat2,binding_d2 = map(flatten_lists, zip(*listoflrpairs2))
        lenbuffer2.extend(lens2)
        lenbuffer_valid2.extend(lens_valid2)
        rewbuffer2.extend(rews2)
        rewbuffer_d_step2.extend(rews_d_step2)
        rewbuffer_d_final2.extend(rews_d_final2)
        rewbuffer_env2.extend(rews_env2)
        rewbuffer_final2.extend(rews_final2)
        rewbuffer_final_stat2.extend(rews_final_stat2)
        classifier_binding2_buffer.extend(binding_d2)

        if writer is not None:
            writer.add_scalar("EpLenMean1", np.mean(lenbuffer1),iters_so_far)
            writer.add_scalar("EpLenValidMean1", np.mean(lenbuffer_valid1),iters_so_far)
            writer.add_scalar("EpRewMean1", np.mean(rewbuffer1),iters_so_far)
            writer.add_scalar("EpRewDStepMean1", np.mean(rewbuffer_d_step1), iters_so_far)
            writer.add_scalar("EpRewDFinalMean1", np.mean(rewbuffer_d_final1), iters_so_far)
            writer.add_scalar("EpRewEnvMean1", np.mean(rewbuffer_env1),iters_so_far)
            writer.add_scalar("EpRewFinalMean1", np.mean(rewbuffer_final1),iters_so_far)
            writer.add_scalar("EpRewFinalStatMean1", np.mean(rewbuffer_final_stat1),iters_so_far)
            writer.add_scalar("EpThisIter1", len(lens1), iters_so_far)
       
            writer.add_scalar("EpLenMean2", np.mean(lenbuffer2),iters_so_far)
            writer.add_scalar("EpLenValidMean2", np.mean(lenbuffer_valid2),iters_so_far)
            writer.add_scalar("EpRewMean2", np.mean(rewbuffer2),iters_so_far)
            writer.add_scalar("EpRewDStepMean2", np.mean(rewbuffer_d_step2), iters_so_far)
            writer.add_scalar("EpRewDFinalMean2", np.mean(rewbuffer_d_final2), iters_so_far)
            writer.add_scalar("EpRewEnvMean2", np.mean(rewbuffer_env2),iters_so_far)
            writer.add_scalar("EpRewFinalMean2", np.mean(rewbuffer_final2),iters_so_far)
            writer.add_scalar("EpRewFinalStatMean2", np.mean(rewbuffer_final_stat2),iters_so_far)
            writer.add_scalar("EpThisIter2", len(lens2), iters_so_far)

            writer.add_scalar("EpRewNetwork", np.mean(classifier_buffer), iters_so_far)
            writer.add_scalar("EpRewAdverse", np.mean(classifier_adverse_buffer), iters_so_far)
            writer.add_scalar("disease_id",disease_id , iters_so_far)  
            writer.add_scalar("EpRewBinding_d1", np.mean(classifier_binding1_buffer), iters_so_far)
            writer.add_scalar("EpRewBinding_d2", np.mean(classifier_binding2_buffer), iters_so_far)

        #disease_count += 1
        assert len(lens1) == len(lens2)
        episodes_so_far += len(lens1)
        timesteps_so_far += min(sum(lens1),sum(lens2))
        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)
        if writer is not None:
            writer.add_scalar("EpisodesSoFar", episodes_so_far, iters_so_far)
            writer.add_scalar("TimestepsSoFar", timesteps_so_far, iters_so_far)
            writer.add_scalar("TimeElapsed", time.time() - tstart, iters_so_far)


        if MPI.COMM_WORLD.Get_rank() == 0:
            with open('molecule_gen/' + args.name_full + '_1.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))
            with open('molecule_gen/' + args.name_full + '_2.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))

            # save
            if iters_so_far % args.save_every == 0:
                fname = './ckpt/' + args.name_full + '_' + str(iters_so_far)
                saver1 = tf.train.Saver(var_list_pi1)
                saver2 = tf.train.Saver(var_list_pi2)
                saver1.save(tf.get_default_session(), fname+'_1')
                saver2.save(tf.get_default_session(), fname+'_2')
                print('model saved!',fname,flush=True)
                # fname = os.path.join(ckpt_dir, task_name)
                # os.makedirs(os.path.dirname(fname), exist_ok=True)
                # saver = tf.train.Saver()
                # saver.save(tf.get_default_session(), fname)
            # if iters_so_far==args.load_step:
        iters_so_far += 1
        counter += 1
        if counter%args.curriculum_step and counter//args.curriculum_step<args.curriculum_num:
            level += 1

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
