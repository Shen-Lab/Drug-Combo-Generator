#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
from tensorboardX import SummaryWriter
import os
import tensorflow as tf

import gym
from gym_molecule.envs.molecule import GraphEnv,get_disease_name_info

def train(args,seed,writer=None):
    from baselines.ppo1 import just_oracle, gcn_policy
    import baselines.common.tf_util as U
    vocab_comp,disease_feat,disease_1hop,disease_1hop_name,disease_gene_list = get_disease_name_info('../../Data/')
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    args.path = os.getcwd()
    print(args.path)
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    if args.env=='molecule':
        env1 = gym.make('molecule-v0')
        env1.init(args.path,vocab_comp,disease_feat,disease_1hop,disease_gene_list,data_type=args.dataset,logp_ratio=args.logp_ratio,qed_ratio=args.qed_ratio,sa_ratio=args.sa_ratio,reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,reward_type=args.reward_type,reward_target=args.reward_target,has_feature=bool(args.has_feature),is_conditional=True,conditional="d1",max_action=args.max_action,min_action=args.min_action) # remember call this after gym.make!!
    elif args.env=='graph':
        env1 = GraphEnv()
        env1.init(reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,dataset=args.dataset) # remember call this after gym.make!!
    print(env1.observation_space)
    def policy_fn(name, ob_space, ac_space,disease_dim):
        return gcn_policy.GCNPolicy(name=name, ob_space=ob_space, ac_space=ac_space, disease_dim=disease_dim,atom_type_num=env1.atom_type_num,args=args)
    env1.seed(workerseed+1)

    if args.env=='molecule':
        env2 = gym.make('molecule-v0')
        env2.init(args.path,vocab_comp,disease_feat,disease_1hop,disease_gene_list,data_type=args.dataset,logp_ratio=args.logp_ratio,qed_ratio=args.qed_ratio,sa_ratio=args.sa_ratio,reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,reward_type=args.reward_type,reward_target=args.reward_target,has_feature=bool(args.has_feature),is_conditional=True,conditional="d2",max_action=args.max_action,min_action=args.min_action) # remember call this after gym.make!!
    elif args.env=='graph':
        env2 = GraphEnv()
        env2.init(reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,dataset=args.dataset) # remember call this after gym.make!!
    print(env2.observation_space)
    env2.seed(workerseed+2)
    just_oracle.deepaffinity(args,env1, env2,policy_fn,
         vocab_comp,disease_feat.shape[0],args.disease_id,disease_1hop_name[args.disease_id],
         max_timesteps=args.num_steps,
         timesteps_per_actorbatch=256,
         clip_param=0.2, entcoeff=0.01,
         optim_epochs=8, optim_stepsize=args.lr, optim_batchsize=32,
         gamma=1, lam=0.95,
        schedule='linear', writer=writer
    )
    env1.close()
    env2.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', type=str, help='environment name: molecule; graph',
                        default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    parser.add_argument('--name', type=str, default='test_conditional')
    parser.add_argument('--name_load', type=str, default='test_conditional')
    # parser.add_argument('--name_load', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    parser.add_argument('--dataset_load', type=str, default='zinc')
    parser.add_argument('--reward_type', type=str, default='gan',help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    parser.add_argument('--reward_target', type=float, default=0.5,help='target reward value')
    parser.add_argument('--logp_ratio', type=float, default=1)
    parser.add_argument('--qed_ratio', type=float, default=1)
    parser.add_argument('--sa_ratio', type=float, default=1)
    parser.add_argument('--gan_step_ratio', type=float, default=1)
    parser.add_argument('--gan_final_ratio', type=float, default=1)
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--has_rl', type=int, default=1)
    # parser.add_argument('--has_expert', type=int, default=1)
    parser.add_argument('--has_d_step', type=int, default=1)
    parser.add_argument('--has_d_final', type=int, default=1)
    parser.add_argument('--has_ppo', type=int, default=1)
    parser.add_argument('--rl_start', type=int, default=250)
    parser.add_argument('--rl_end', type=int, default=int(1e6))
    parser.add_argument('--expert_start', type=int, default=0)
    parser.add_argument('--expert_end', type=int, default=int(1e6))
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--load', type=int, default=1)
    parser.add_argument('--load_step', type=int, default=6800)
    # parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--curriculum_num', type=int, default=6)
    parser.add_argument('--curriculum_step', type=int, default=200)
    parser.add_argument('--supervise_time', type=int, default=4)
    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--layer_num_g', type=int, default=3)
    parser.add_argument('--layer_num_d', type=int, default=3)
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--stop_shift', type=int, default=-3)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_concat', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=1)
    parser.add_argument('--emb_size', type=int, default=64) # default 64
    parser.add_argument('--gcn_aggregate', type=str, default='mean')# sum, mean, concat
    parser.add_argument('--gan_type', type=str, default='wgan')
    parser.add_argument('--gate_sum_d', type=int, default=0)
    parser.add_argument('--mask_null', type=int, default=0)
    parser.add_argument('--is_conditional', type=int, default=1) # default 0
    parser.add_argument('--conditional', type=str, default='d1') # default 0
    parser.add_argument('--max_action', type=int, default=128) # default 0
    parser.add_argument('--min_action', type=int, default=20) # default 0
    parser.add_argument('--bn', type=int, default=0)
    parser.add_argument('--name_full',type=str,default='')
    parser.add_argument('--name_full_load',type=str,default='')
 
    parser.add_argument('--disease_id',type=int,default=200)
    parser.add_argument('--network_weight',type=int,default=10)
    parser.add_argument('--deepaffinity_thr',type=int,default=6)
    parser.add_argument('--others_weight',type=int,default=1)
    parser.add_argument('--path',type=str,default=os.getcwd())
    return parser

def main():
    args = molecule_arg_parser().parse_args()
    args.name_full = args.env + '_' + args.dataset + '_' + args.name
    args.name_full_load1 = args.env + '_' + args.dataset_load + '_' + args.name_load + '_' + str(args.load_step) + '_1'
    args.name_full_load2 = args.env + '_' + args.dataset_load + '_' + args.name_load + '_' + str(args.load_step) + '_2'
    print(args)

    train(args,seed=args.seed,writer=None)

if __name__ == '__main__':
    main()
