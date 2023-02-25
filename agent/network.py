import torch
import torch.nn as nn
import math


def orthogonal_init(layer, gain=math.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


class Actor_Critic(nn.Module):
    def __init__(self, config):
        super(Actor_Critic, self).__init__()
        self.map_layer = nn.Sequential(
            orthogonal_init(nn.Conv2d(config.s_map_dim[0], 8, kernel_size=5, stride=1, padding=2)),  # s_map_dim[0]*24*24->8*24*24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8*24*24->8*12*12

            orthogonal_init(nn.Conv2d(8, 16, kernel_size=3, stride=1)),  # 8*12*12->16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16*10*10->16*5*5

            nn.Flatten(),
            orthogonal_init(nn.Linear(16 * 5 * 5, config.hidden_dim)),
            nn.ReLU(),
        )

        self.sensor_layer = nn.Sequential(
            orthogonal_init(nn.Linear(config.s_sensor_dim[0], config.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, config.action_dim), gain=0.01)  # hidden_dim*2->action_dim
        self.critic_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, 1), gain=1.0)  # hidden_dim*2->1

    def get_feature(self, s_map, s_sensor):
        s_map = s_map.float() / 255.0
        s_map = self.map_layer(s_map)

        s_sensor = self.sensor_layer(s_sensor)
        feature = torch.cat([s_map, s_sensor], dim=-1)

        return feature

    def get_logit_and_value(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        logit = self.actor_out(feature)
        value = self.critic_out(feature)

        return logit, value.squeeze(-1)

    def actor(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        logit = self.actor_out(feature)

        return logit

    def critic(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        value = self.critic_out(feature)

        return value.squeeze(-1)


