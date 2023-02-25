import argparse
from utils.str2bool import str2bool


def get_configs():
    parser = argparse.ArgumentParser()
    # ---------------------------------------------------Training-------------------------------------------------
    parser.add_argument("--env_version", type=str, default="v1", help="The version of gym env")
    parser.add_argument("--random_obstacle", type=str2bool, default=True, help="Whether to use random obstacles")
    parser.add_argument("--num_envs_per_map", type=int, default=4, help="The number of vectorized envs per map")
    parser.add_argument("--number", type=int, default=1, help="Experiment index")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device")
    parser.add_argument("--evaluate_freq", type=int, default=int(1e3), help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--save_model", type=str2bool, default=True, help="Whether to save models")
    # ---------------------------------------------------PPO---------------------------------------------------
    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help=" Maximum number of training steps")
    parser.add_argument("--rollout_steps", type=int, default=256, help="Rollout steps")
    parser.add_argument("--minibatches", type=int, default=8, help="The number of minibatches")
    parser.add_argument("--hidden_dim", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=str2bool, default=True, help="Advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--use_grad_clip", type=str2bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--critic_loss_coef", type=float, default=1.0, help="Critic loss coefficient")
    parser.add_argument("--use_AdamW", type=str2bool, default=True, help="Whether to use AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight_decay in AdamW")
    parser.add_argument("--use_DrAC", type=str2bool, default=True, help="Whether to use DrAC")
    parser.add_argument("--aug_coef", type=float, default=0.1, help="Coefficient in DrAC")

    config = parser.parse_args()

    return config


