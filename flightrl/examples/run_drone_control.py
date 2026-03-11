#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import tensorflow as tf

#
from stable_baselines import logger

#
from rpg_baselines.common.policies import MlpPolicy
from rpg_baselines.ppo.ppo2_per import PPO2withPER
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./examples/saved/2025-10-28-14-53-02.zip',
                        help='trained weight path')
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help="PER prioritization exponent alpha")
    parser.add_argument('--per_beta0', type=float, default=0.4,
                        help="Initial PER IS correction beta")
    parser.add_argument('--per_beta_increment', type=float, default=0.001,
                        help="PER beta increment per PPO update")
    parser.add_argument('--per_epsilon', type=float, default=1e-6,
                        help="PER epsilon to avoid zero priorities")
    parser.add_argument('--per_capacity', type=int, default=200000,
                        help="PER replay capacity")
    parser.add_argument('--total_timesteps', type=int, default=50000000,
                        help="Total environment timesteps for training")
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 100
        cfg["env"]["num_threads"] = 10

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1())
    yaml = YAML(typ='unsafe', pure=True)
    import io
    stream = io.StringIO()
    yaml.dump(cfg, stream)
    config_str = stream.getvalue()

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        # Use a slightly deeper network for this complex gate navigation task
        model = PPO2withPER(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,
            policy_kwargs=dict(
                # Deeper network with more capacity for the complex racing task
                net_arch=[dict(pi=[128, 128], vf=[128, 128])], 
                act_fun=tf.nn.tanh),
            env=env,
            lam=0.95,
            gamma=0.999,  # Higher discount factor to value future rewards more
            n_steps=512,  # Longer horizon for better learning
            ent_coef=0.02,  # Keep same exploration coefficient
            learning_rate=3e-4,
            vf_coef=0.6,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.3,
            per_alpha=args.per_alpha,
            per_beta=args.per_beta0,
            per_beta_increment=args.per_beta_increment,
            per_epsilon=args.per_epsilon,
            per_capacity=args.per_capacity,
            verbose=1,
        )

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(args.total_timesteps),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO2withPER.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
