import torch


class PPO_Buffer:
    def __init__(self, config, num_envs):
        self.device = config.device
        self.gamma = config.gamma
        self.lamda = config.lamda
        self.s_map_dim = config.s_map_dim
        self.s_sensor_dim = config.s_sensor_dim
        self.rollout_steps = config.rollout_steps
        self.num_envs = num_envs
        self.use_adv_norm = config.use_adv_norm
        self.buffer = {'s_map': torch.zeros(((self.rollout_steps, self.num_envs,) + self.s_map_dim), dtype=torch.uint8, device=self.device),
                       's_sensor': torch.zeros(((self.rollout_steps, self.num_envs,) + self.s_sensor_dim), dtype=torch.float32, device=self.device),
                       'value': torch.zeros((self.rollout_steps + 1, self.num_envs), dtype=torch.float32, device=self.device),
                       'a': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.int, device=self.device),
                       'logprob': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device),
                       'r': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device),
                       'terminal': torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device),
                       }
        self.count = 0

    def store_transition(self, s, value, a, logprob, r, terminal):
        self.buffer['s_map'][self.count] = s['s_map']
        self.buffer['s_sensor'][self.count] = s['s_sensor']
        self.buffer['value'][self.count] = value
        self.buffer['a'][self.count] = a
        self.buffer['logprob'][self.count] = logprob
        self.buffer['r'][self.count] = torch.tensor(r, dtype=torch.float32, device=self.device)
        self.buffer['terminal'][self.count] = torch.tensor(terminal, dtype=torch.float32, device=self.device)
        self.count += 1

    def store_value(self, value):
        self.buffer['value'][self.count] = value

    def get_adv(self):  # Use GAE to calculate advantage and v_target
        with torch.no_grad():  # adv and v_target have no gradient
            # self.buffer['value'][:-1]=v(s)
            # self.buffer['value'][1:]=v(s')
            deltas = self.buffer['r'] + self.gamma * (1.0 - self.buffer['terminal']) * self.buffer['value'][1:] - self.buffer['value'][:-1]  # deltas.shape(rollout_steps,num_envs)
            adv = torch.zeros_like(self.buffer['r'], device=self.device)  # adv.shape(rollout_steps,num_envs)
            gae = 0
            for t in reversed(range(self.rollout_steps)):
                gae = deltas[t] + self.gamma * self.lamda * gae * (1.0 - self.buffer['terminal'][t])
                adv[t] = gae
            v_target = adv + self.buffer['value'][:-1]
            if self.use_adv_norm:  # Advantage normalization
                adv = ((adv - torch.mean(adv)) / (torch.std(adv) + 1e-5))

        return adv, v_target

    def get_training_data(self):
        adv, v_target = self.get_adv()
        # batch_size = rollout_steps * num_envs
        batch = {'s_map': self.buffer['s_map'].reshape((-1,) + self.s_map_dim),  # (batch_size,s_map_dim)
                 's_sensor': self.buffer['s_sensor'].reshape((-1,) + self.s_sensor_dim),  # (batch_size,s_sensor_dim)
                 'a': self.buffer['a'].reshape(-1),  # (batch_size,)
                 'logprob': self.buffer['logprob'].reshape(-1),  # (batch_size,)
                 'adv': adv.reshape(-1),  # (batch_size,)
                 'v_target': v_target.reshape(-1),  # (batch_size,)
                 }
        return batch
