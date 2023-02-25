import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import grid_simulator
import gym
import time
from agent.ppo_discrete import PPO_Discrete
from configs import get_configs


class Runner:
    def __init__(self, config):
        self.algorithm = config.algorithm
        self.env_version = config.env_version
        self.random_obstacle = config.random_obstacle
        self.num_envs_per_map = config.num_envs_per_map
        self.number = config.number
        self.seed = config.seed
        self.max_train_steps = config.max_train_steps
        self.evaluate_freq = config.evaluate_freq
        self.evaluate_times = config.evaluate_times
        self.save_model = config.save_model
        self.rollout_steps = config.rollout_steps

        # Set seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Create env
        self.map_names = ["train_map_l1", "train_map_l2", "train_map_l3", "train_map_l4", "train_map_l5"]

        self.map_index_now = 0
        vector_maps = [lambda: gym.make("ExploreEnv-{}".format(self.env_version), map_name=self.map_names[self.map_index_now], random_obstacle=self.random_obstacle, training=True)] * self.num_envs_per_map
        self.envs = gym.vector.AsyncVectorEnv(vector_maps)
        self.env_evaluate = [gym.make('ExploreEnv-{}'.format(self.env_version), map_name=self.map_names[self.map_index_now], random_obstacle=self.random_obstacle, training=False)]
        self.envs.reset(seed=self.seed)
        for env in self.env_evaluate:
            env.reset(seed=self.seed)

        config.s_map_dim = self.envs.single_observation_space["s_map"].shape
        config.s_sensor_dim = self.envs.single_observation_space["s_sensor"].shape
        config.action_dim = self.envs.single_action_space.n
        print("s_map_dim={}".format(config.s_map_dim))
        print("s_sensor_dim={}".format(config.s_sensor_dim))
        print("action_dim={}".format(config.action_dim))
        self.num_envs = len(vector_maps)
        print("num_envs={}".format(self.num_envs))

        # Create agent
        self.agent = PPO_Discrete(config, num_envs=self.num_envs)
        # Create tensorboard
        self.writer = SummaryWriter(log_dir='./tensorboard/{}_env_{}_number_{}_seed_{}'.format(self.algorithm, self.env_version, self.number, self.seed))

        self.step_times = 0  # Total vectorized steps of envs
        self.total_steps_per_map = np.zeros(self.map_index_now + 1, dtype=np.int32)  # Total transitions sample on per map
        self.delta_steps = np.ones(self.map_index_now + 1, dtype=np.int32) * self.num_envs_per_map
        self.evaluate_num = -1  # Record the number of evaluations
        self.evaluate_rewards = [[] for _ in range(len(self.map_names))]
        self.evaluate_explore_rates = [[] for _ in range(len(self.map_names))]
        self.map_explore_rates = []  # Record the map exploration rate
        self.switch_index = []

    def switch_map(self):
        self.map_index_now += 1
        vector_maps = []
        for map_index in range(self.map_index_now + 1):
            vector_maps += [lambda index=map_index: gym.make("ExploreEnv-{}".format(self.env_version), map_name=self.map_names[index], random_obstacle=self.random_obstacle, training=True)] * self.num_envs_per_map
        self.envs = gym.vector.AsyncVectorEnv(vector_maps)
        self.env_evaluate.append(gym.make('ExploreEnv-{}'.format(self.env_version), map_name=self.map_names[self.map_index_now], random_obstacle=self.random_obstacle, training=False))
        s, info = self.envs.reset(seed=self.seed)
        self.env_evaluate[-1].reset(seed=self.seed)

        self.map_explore_rates = []
        self.switch_index.append(self.evaluate_num)
        self.total_steps_per_map = np.concatenate([self.total_steps_per_map, [0]])
        self.delta_steps = np.ones(self.map_index_now + 1, dtype=np.int32) * self.num_envs_per_map
        self.num_envs = len(vector_maps)
        print("num_envs={}".format(self.num_envs))
        self.agent.reset_buffer(num_envs=self.num_envs)  # Reset buffer
        print("Add {}".format(self.map_names[self.map_index_now]))
        self.evaluate_single_map(self.map_index_now, self.env_evaluate[-1], self.total_steps_per_map[self.map_index_now])

        return s

    def run(self, ):
        s, info = self.envs.reset()
        while self.step_times < self.max_train_steps:
            if self.step_times // self.evaluate_freq > self.evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                if len(self.map_explore_rates) >= 10 and np.mean(self.map_explore_rates[-10:]) >= 0.95 and self.map_index_now < len(self.map_names) - 1:
                    s = self.switch_map()

            a, logprob, value, s = self.agent.get_action_and_value(s)
            s_, r, terminal, _, info = self.envs.step(a.cpu().numpy())
            self.step_times += 1
            self.total_steps_per_map += self.delta_steps

            # Store the transition
            self.agent.ppo_buffer.store_transition(s, value, a, logprob, r, terminal)
            s = s_

            if self.agent.ppo_buffer.count == self.rollout_steps:
                value = self.agent.get_value(s)
                self.agent.ppo_buffer.store_value(value)
                self.agent.update() # Update
                self.agent.ppo_buffer.count = 0

        self.evaluate_policy()

        # Save reward and map exploration rate
        for map_index in range(len(self.map_names)):
            np.save('./data_train/{}_env_{}_{}_number_{}_seed_{}_reward.npy'.format(self.algorithm, self.env_version, self.map_names[map_index], self.number, self.seed), np.array(self.evaluate_rewards[map_index]))
            np.save('./data_train/{}_env_{}_{}_number_{}_seed_{}_rate.npy'.format(self.algorithm, self.env_version, self.map_names[map_index], self.number, self.seed), np.array(self.evaluate_explore_rates[map_index]))
        np.save('./data_train/{}_env_{}_number_{}_seed_{}_switch_index.npy'.format(self.algorithm, self.env_version, self.number, self.seed), np.array(self.switch_index))
        time.sleep(1.0)
        print("Successfully save reward and rate")

    def evaluate_policy(self):
        self.evaluate_num += 1
        for map_index, env in enumerate(self.env_evaluate):  # The policy is evaluated simultaneously on all previously experienced levels
            self.evaluate_single_map(map_index, env, self.total_steps_per_map[map_index])
        if self.save_model:
            self.agent.save_model(algorithm=self.algorithm, env_version=self.env_version, map_name='all_maps', number=self.number, seed=self.seed, index=self.evaluate_num)

    def evaluate_single_map(self, map_index, env, map_steps):
        evaluate_reward = 0
        evaluate_explore_rate = 0
        for _ in range(self.evaluate_times):
            s, info = env.reset()
            terminal, episode_reward = False, 0
            while not terminal:
                a = self.agent.evaluate_policy(s)
                s_, r, terminal, _, info = env.step(a)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            evaluate_explore_rate += info['explore_rate']
        evaluate_reward /= self.evaluate_times
        evaluate_explore_rate /= self.evaluate_times
        self.evaluate_rewards[map_index].append(evaluate_reward)
        self.evaluate_explore_rates[map_index].append(evaluate_explore_rate)
        self.writer.add_scalar('step_rewards_{}'.format(self.map_names[map_index]), evaluate_reward, global_step=map_steps)
        self.writer.add_scalar('step_explore_rate_{}'.format(self.map_names[map_index]), evaluate_explore_rate, global_step=map_steps)
        print("map_steps:{} \t map_name:{} \t reward:{} \t explore_rate:{}".format(map_steps, self.map_names[map_index], evaluate_reward, evaluate_explore_rate))
        if map_index == self.map_index_now:
            self.map_explore_rates.append(evaluate_explore_rate)


if __name__ == '__main__':
    config = get_configs()
    config.algorithm = "CCPPO"

    config_dict = vars(config)  # print all hyper-parameters
    for key in config_dict.keys():
        print("{}={}".format(key, config_dict[key]))

    runner = Runner(config=config)
    runner.run()
