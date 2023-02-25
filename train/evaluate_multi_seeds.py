import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import numpy as np
import argparse
from agent.network import Actor_Critic
import grid_simulator
import gym
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils.str2bool import str2bool
from torch.distributions import Categorical

abspath = os.path.dirname(os.path.abspath(__file__))

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.algorithm = config.algorithm
        self.env_version = config.env_version
        self.random_obstacle = config.random_obstacle
        self.train_map_name = config.train_map_name
        self.test_map_name = config.test_map_name
        self.number = config.number
        self.seed = config.seed
        self.model_index = config.model_index
        self.device = config.device

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Create env
        self.env = gym.make('ExploreEnv-{}'.format(self.env_version), map_name=self.test_map_name, random_obstacle=self.random_obstacle, training=False, render=False)
        self.env.reset(seed=self.seed)
        config.s_map_dim = self.env.observation_space["s_map"].shape
        config.s_sensor_dim = self.env.observation_space["s_sensor"].shape
        config.action_dim = self.env.action_space.n

        self.net = Actor_Critic(config).to(self.device)

        model_path = abspath+"/model/{}_env_{}_{}_number_{}_seed_{}_index_{}.pth".format(self.algorithm, self.env_version, self.train_map_name, self.number, self.seed, self.model_index)
        if self.model_index:
            print("load model...")
            self.net.load_state_dict(torch.load(model_path))
        else:
            print("model index=0")
        print("model_path={}".format(model_path))
        print("test_map_name={}".format(self.test_map_name))

    def evaluate(self, ):
        evaluate_explore_rate = []
        steps_95 = []
        for evaluate_time in range(self.config.evaluate_times):
            s, info = self.env.reset()
            done = False
            win_95 = False
            while not done:
                a = self.choose_action(s)
                s_, r, done, _, info = self.env.step(a)
                s = s_
                if info['explore_rate'] >= 0.95 and not win_95:
                    steps_95.append(info['episode_steps'])
                    win_95 = True
            explore_rate = info['explore_rate']
            evaluate_explore_rate.append(explore_rate)

        return evaluate_explore_rate, steps_95

    def choose_action(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).unsqueeze(0).to(self.device)
            s_sensor = torch.from_numpy(s['s_sensor']).unsqueeze(0).to(self.device)
            logit = self.net.actor(s_map, s_sensor)
            a = Categorical(logits=logit).sample()
            return a.cpu().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--algorithm", type=str, default='CCPPO', help=" The name of the algorithm")
    parser.add_argument("--env_version", type=str, default="v1", help="env_version")
    parser.add_argument("--train_map_name", type=str, default="all_maps", help="train_map_name")
    parser.add_argument("--test_map_name", type=str, default="test_map_l5", help="test_map_name")
    parser.add_argument("--random_obstacle", type=str2bool, default=True, help="test_map_name")
    parser.add_argument("--number", type=int, default=1, help="number")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")
    parser.add_argument("--evaluate_times", type=int, default=20, help="Evaluate times")
    parser.add_argument("--model_index", type=int, default=300, help="model_index")
    parser.add_argument("--hidden_dim", type=int, default=32, help="The number of neurons in hidden layers of the neural network")

    config = parser.parse_args()

    config_dict = vars(config)
    for key in config_dict.keys():
        print("{}={}".format(key, config_dict[key]))

    for test_map_name in ['test_map_l1', 'test_map_l2', 'test_map_l3', 'test_map_l4', 'test_map_l5']:
        rate_list = []
        steps_list_95 = []

        for seed in [1, 11, 30, 80, 90]:
            config.test_map_name = test_map_name
            config.seed = seed
            evaluator = Evaluator(config)
            rate,steps_95 = evaluator.evaluate()
            rate_list.append(rate)
            steps_list_95.append(steps_95)

        rate_list = np.concatenate(rate_list)
        print(rate_list.shape)
        steps_list_95 = np.concatenate(steps_list_95)

        rate_mean = np.round(np.mean(rate_list), 3)
        rate_std = np.round(np.std(rate_list), 3)
        print("map:{} \trate_mean:{} \t rate_std:{}".format(test_map_name, rate_mean, rate_std))
        print("steps_95_mean:{} \t steps_95_std:{} \t".format(np.round(np.mean(steps_list_95), 3), np.round(np.std(steps_list_95), 3)))

        print("\n")
