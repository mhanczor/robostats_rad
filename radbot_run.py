#!/usr/bin/env python
import os, logging, gym
import tensorflow as tf
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import Model, Runner
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
   def make_env(rank):
       env = gym.make(env_id)
       env.seed(seed + rank)
    #    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    #    gym.logger.setLevel(logging.WARN)
       return env

   set_global_seeds(seed)
   env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

   if policy == 'cnn':
       policy_fn = CnnPolicy
   elif policy == 'lstm':
       policy_fn = LstmPolicy
   elif policy == 'radlstm':
       policy_fn = RadLstmPolicy

   learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
   env.close()


def load_model(load_path, policy, env_id, seed=0, num_procs=1, nenvs=1, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-3, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):


    env = gym.make(env_id)
    env.seed(seed)
#        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
#    gym.logger.setLevel(logging.WARN)

    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    env.num_envs = 1
    tf.reset_default_graph()
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    model.load(load_path)

    return env, runner

def play(env_id, num_timesteps, policy, load_path=None):

    env, runner = load_model(load_path, policy, env_id)
    env.reset()

    for _ in range(num_timesteps):
        env.render()
        obs, states, rewards, masks, actions, values = runner.run()

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='RadRoomSimple-v0')
    parser.add_argument('--filepath', help='Model Filepath', default='/home/hades/Documents/Robo_Stats/Final_Project/radbot_gym/saved_models/atari_saved.model')
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='lstm')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--mode', help='Mode', choices=['train', 'play'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.env, args.num_timesteps, seed=0, policy=args.policy, lrschedule=args.lrschedule, num_cpu=1)
    elif args.mode == 'play':
        play(args.env, args.num_timesteps, policy=args.policy, load_path=args.filepath)

#    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
#        policy=args.policy, lrschedule=args.lrschedule, num_cpu=8)

if __name__ == '__main__':
    main()
