from copy import deepcopy
import torch


def rotation_s(s_map, s_sensor, k):  # Clockwise rotation 90,180,210
    aug_s_map = torch.rot90(s_map, k=-k, dims=[2, 3])
    aug_s_sensor = deepcopy(s_sensor)
    aug_s_sensor[:, -1] = (aug_s_sensor[:, -1] + 0.25 * k) % 1  # The steering angle is modified accordingly
    return aug_s_map, aug_s_sensor
