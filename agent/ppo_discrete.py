import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import numpy as np
from .network import Actor_Critic
from .buffer import PPO_Buffer
from utils.data_augmentation import rotation_s



class PPO_Discrete:
    def __init__(self, config, num_envs):
        self.config = config
        self.device = config.device
        self.minibatches = config.minibatches
        self.max_train_steps = config.max_train_steps
        self.lr = config.lr  # Learning rate of actor
        self.gamma = config.gamma  # Discount factor
        self.lamda = config.lamda  # GAE parameter
        self.epsilon = config.epsilon  # PPO clip parameter
        self.K_epochs = config.K_epochs  # PPO parameter
        self.entropy_coef = config.entropy_coef  # Entropy coefficient
        self.critic_loss_coef = config.critic_loss_coef
        self.set_adam_eps = config.set_adam_eps
        self.use_grad_clip = config.use_grad_clip
        self.use_adv_norm = config.use_adv_norm
        self.use_AdamW = config.use_AdamW
        self.weight_decay = config.weight_decay
        self.use_DrAC = config.use_DrAC
        self.aug_coef = config.aug_coef

        self.ppo_buffer = PPO_Buffer(config, num_envs)
        self.ac_net = Actor_Critic(config).to(self.device)

        if self.set_adam_eps:  # Set Adam epsilon=1e-5
            if self.use_AdamW:
                self.optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.lr, eps=1e-5, weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr, eps=1e-5)
        else:
            if self.use_AdamW:
                self.optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=self.lr)

    def evaluate_policy(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).unsqueeze(0).to(self.device)
            s_sensor = torch.from_numpy(s['s_sensor']).unsqueeze(0).to(self.device)

            logit = self.ac_net.actor(s_map, s_sensor)
            dist = Categorical(logits=logit)
            a = dist.sample()
            return a.cpu().item()

    def get_value(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).to(self.device)
            s_sensor = torch.from_numpy(s['s_sensor']).to(self.device)
            value = self.ac_net.critic(s_map, s_sensor)
            return value

    def get_action_and_value(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).to(self.device)
            s_sensor = torch.from_numpy(s['s_sensor']).to(self.device)

            logit, value = self.ac_net.get_logit_and_value(s_map, s_sensor)
            dist = Categorical(logits=logit)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a, logprob, value, {'s_map': s_map, 's_sensor': s_sensor}

    def update(self, ):
        batch = self.ppo_buffer.get_training_data()  # Get training data
        batch_size = batch['a'].shape[0]
        minibatch_size = batch_size // self.minibatches
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, False):
                logits_now, values_now = self.ac_net.get_logit_and_value(batch['s_map'][index], batch['s_sensor'][index])
                dist_now = Categorical(logits=logits_now)
                dist_entropy = dist_now.entropy()  # shape(minibatch_size,)
                entropy_loss = dist_entropy.mean()

                logprob_now = dist_now.log_prob(batch['a'][index])  # shape(minibatch_size,)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(logprob_now - batch['logprob'][index])  # shape(minibatch_size,)
                surr1 = ratios * batch['adv'][index]  # Only calculate the gradient of 'logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * F.mse_loss(batch['v_target'][index], values_now)

                ppo_loss = actor_loss - self.entropy_coef * entropy_loss + self.critic_loss_coef * critic_loss

                if self.use_DrAC:
                    aug_s_map, aug_s_sensor = rotation_s(s_map=batch['s_map'][index], s_sensor=batch['s_sensor'][index], k=np.random.randint(1, 4))
                    aug_logits, aug_values = self.ac_net.get_logit_and_value(aug_s_map, aug_s_sensor)

                    aug_actor_loss = F.kl_div(input=torch.log_softmax(aug_logits, dim=-1), target=torch.softmax(logits_now, dim=-1).detach(), reduction='batchmean')
                    aug_critic_loss = 0.5 * F.mse_loss(values_now.detach(), aug_values)

                    loss = ppo_loss + (aug_actor_loss + aug_critic_loss) * self.aug_coef
                else:
                    loss = ppo_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:  # Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
                self.optimizer.step()

    def reset_buffer(self, num_envs):
        self.ppo_buffer = PPO_Buffer(config=self.config, num_envs=num_envs)

    def save_model(self, algorithm, env_version, map_name, number, seed, index):
        torch.save(self.ac_net.state_dict(), "./model/{}_env_{}_{}_number_{}_seed_{}_index_{}.pth".format(algorithm, env_version, map_name, number, seed, index))
