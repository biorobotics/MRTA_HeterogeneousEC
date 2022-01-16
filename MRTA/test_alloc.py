'''
Author: Jiaheng Hu
Test the trained network
'''

import torch
from .utils import generate_true_data, calc_gradient_penalty, int_to_onehot, calc_reward_from_rnet
from .params import get_params
from .Networks.Generator import AllocationGenerator
from .Networks.Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
from .MAETF.simulator import MultiAgentEnv
import numpy as np
import os
from collections import defaultdict
import torch.nn.functional as F
from .Networks.RewardNet import RewardNet

def test():
    params = get_params()

    if params['sim_env']:
        print("Testing in simulation environment")
    else:
        print("Testing in toy enviroment")

    batch_size = params['batch_size']
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'],
                        agent_num=params['agent_num'])
    print("Number of grids: ", params['env_grid_num'])
    print("Number of agent types: ", params['n_agent_types'])
    print("Number of environment types: ", params['n_env_types'])
    print("Number of agents: ", params['agent_num'])


    worker_device = torch.device("cuda:0")

    # Models
    generator = AllocationGenerator(
        n_agent=params['n_agent_types'],
        n_env_grid=params['env_grid_num'],
        env_input_len=params['env_input_len'],
        design_input_len=params['design_input_len'],
        norm=params['gen_norm'],
        layer_size=params['design_layer_size']).to(worker_device)

    generator.load_state_dict(torch.load(os.path.join(params['test_loc'], "generator_weight")))
    print("Loaded state dictionary")
    generator.eval()

    reward_net = RewardNet(params['n_agent_types'],
                           env_length=params['n_env_types'],
                           norm=params['reward_norm'],
                           n_hidden_layers=5, hidden_layer_size=256).to(worker_device)
    reward_net.load_state_dict(torch.load(params['regress_net_loc']))
    reward_net.eval()

    sample_size = 1000
    # env_type = [0, 1, 2, 3]
    # env_type = [1, 2, 3, 0]
    # env_type = [3, 2, 2, 1]
    env_type = [0,3]
    # [2, 3, 0, 3]
    env_onehot = torch.tensor(int_to_onehot(env_type, params['n_env_types']),
                              dtype=torch.float, device=worker_device)
    env_onehot = env_onehot.reshape(1, -1).repeat(sample_size, 1)
    # print("Env one shot generated: ", env_onehot)
    noise = torch.normal(0, 1, size=(sample_size, params['design_input_len']), device=worker_device)

    import time
    start_time = time.time()
    generated_data_logits = generator(noise, env_onehot)
    generated_data_raw = F.softmax(generated_data_logits, dim=-1)
    generated_data_raw = generated_data_raw.detach().cpu().numpy().astype(float)
    # print("Generated allocation: ", generated_data_raw)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(f"env type is: {env_type}")


    int_alloc = np.array([env.get_integer(alloc) for alloc in generated_data_raw])
    print("Int alloc: ", int_alloc)
    rewards = calc_reward_from_rnet(env, reward_net, int_alloc, env_onehot, sample_size)
    # print("Rewards earned for allocation: ", rewards)
    # When the environment type is sim_env, then the reward is ergodicity, so we use argmin(rewards) to find the best allocation
    print("Best ergodicity is for: ", torch.argmin(rewards))
    print("Allocation with best ergodicity: ", int_alloc[torch.argmin(rewards)])

    sorted, indices = torch.sort(rewards, descending=True)
    for i in range(5):
        print(sorted[i])
        print(int_alloc[indices[i]])
    from .evo_function import how_many_goptima
    radius, fitness_goptima, accuracy = 10, sorted[0], 0.0000005
    print(how_many_goptima(int_alloc.reshape(sample_size, 6),rewards.cpu().numpy(), radius, fitness_goptima, accuracy )[0])
    return int_alloc[torch.argmin(rewards)]


def get_count_dict(generated_data_raw, env, env_type):
    int_alloc = [env.get_integer(alloc) for alloc in generated_data_raw]
    def default():
        return 0
    dict = defaultdict(default)
    for alloc in int_alloc:
        assignment_str = alloc.tostring()
        dict[assignment_str] += 1
    sorted_key = sorted(dict, key=dict.get)
    for i in range(10):
        max_assign = np.fromstring(sorted_key[-i], dtype=float)
        print(max_assign.reshape((3,4)))

if __name__ == '__main__':
    best_alloc = test()