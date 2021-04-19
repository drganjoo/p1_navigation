import numpy as np
import replay
import torch
import time
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, NamedTuple
from replay import AnnealingRate, PrioritizeExperience, ReplayBufferWithPriority

from model import QNetwork

Action = int

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LR = 5e-4
GAMMA = 0.99
TAU = 1e-3 
EPS_START = 0.98
EPS_END = 0.01
EPS_DECAY_START = 20_000
EPS_DECAY_END = 1_00_000
MIN_EXPERIENCE_BEFORE_LEARN = 10_000
LEARN_EVERY_N_STEPS = 4
UPDATE_EVERY_N_STEPS = 12

parts = torch.__version__.split('.')
torch_latest = parts[0] == '1' and parts[1] >= '5'

class BaseAgent:
    """
    All agents need to support at least the following:
        act
        step
    """
    def __init__(self, action_space : int):
        self.action_space = action_space

    def act(self, state : np.ndarray) -> Action:
        raise "base agent does not have the capability to act"

    def step(self, state : np.ndarray, action : int, reward : float, next_state : np.ndarray, done : bool) -> Action:
        raise "base agent does not have the capability to learn"

class DDQNAgent(BaseAgent):
    def __init__(self,
                state_space : int,
                action_space: int,
                device: str,
                greedy_epsilon: replay.AnnealingRate = None,
                seed: float = None):
        # initialize the base agent
        super().__init__(action_space)

        if greedy_epsilon == None:
            self.eps = AnnealingRate(EPS_START, EPS_END, EPS_DECAY_START, EPS_DECAY_END)

        self.replay_buffer = ReplayBufferWithPriority(BUFFER_SIZE, BATCH_SIZE, seed)
        self.device = device

        # target and local networks
        self.qtarget_network = QNetwork(state_space, action_space, seed).to(device)
        self.qlocal_network = QNetwork(state_space, action_space, seed).to(device)

        # qtarget is never learnt, always evaluated and copied into
        self.qtarget_network.eval()

        # optimizer for learning
        self.optimizer = optim.Adam(self.qlocal_network.parameters(), lr = LR)

        # all new experiences that are put in the replay buffer have to be picked
        # at least once. Therefore their priority is kept at maximum. When the agent
        # learns it remembers the max priority seen and then uses that next time for new experiences
        self.max_priority = 1

        self.step_count = 0
        self.debug = False
        self.is_training = True

        if seed != None:
            np.random.seed(seed)

    def _choose_best_action(self, state : np.ndarray) -> Action:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.qlocal_network.eval()
            action_values = self.qlocal_network(state).to('cpu')
            self.qlocal_network.train()
        return int(np.argmax(action_values.numpy()))

    def act(self, state : np.ndarray) -> Action:
        if self.is_training and np.random.random() < self.eps.get_current():
            return np.random.randint(self.action_space)
        else:
            return self._choose_best_action(state)

    def step(self, state : np.ndarray, action : int, reward : float, next_state : np.ndarray, done : bool):
        assert self.is_training == True, "Step should not be called when in evaluation mode"

        # assign the highest priority to the new incoming experience
        # to make sure it is picked at least once
        self.replay_buffer.add_with_max(state, action, reward, next_state, done) 
        
        # cannot learn before we have enough experiences in the replay buffer
        self.step_count += 1
        if len(self.replay_buffer) < MIN_EXPERIENCE_BEFORE_LEARN:
            return
        
        if self.step_count % LEARN_EVERY_N_STEPS  == 0:
            self._learn_from_replay()

        if self.step_count % UPDATE_EVERY_N_STEPS == 0:
            self._soft_update(self.qlocal_network, self.qtarget_network, TAU)

    def _learn_from_replay(self):
        print('\rlearn_from_replay', end='')

        indices_np, states_np, actions_np, rewards_np, next_states_np, dones_np, weights_np = self.replay_buffer.sample_batch()

        states = torch.from_numpy(states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        dones = torch.from_numpy(dones_np).float().to(self.device)
        weights = torch.from_numpy(weights_np).float().to(self.device)

        # loss function: Q(s) - Q*(s)

        # Q(s, a)
        # forward will give values for all actions in that state, we only need to 
        # select the ath action (which was chosen at experience e)'s value
        q_s_local = self.qlocal_network(states).gather(1, actions)

        # V(s,a) = r + gamma * max_a Q(s', a)
        with torch.no_grad():
            q_s_prime_target = self.qtarget_network(next_states).detach()

            # put the local network in eval mode so that in future if it has dropout layers etc.
            # they are used in eval
            self.qlocal_network.eval()
            q_s_prime_local = self.qlocal_network(next_states).detach()
            self.qlocal_network.train()

            # choose what action is max from local network
            a_s_prime_local = torch.argmax(q_s_prime_local, 1)

            # get action value from the target network for the indices chosen from local
            v_s_prime_target = q_s_prime_target.gather(1, a_s_prime_local.unsqueeze(1))

            # delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
            # dones is a column vector, we need to convert it to row vector to multiply
            # it with gamma * max_a.values
            dones_row = (1 - dones).unsqueeze(1)        # a column vector
            future_rewards = dones_row * GAMMA * v_s_prime_target
            
            y = rewards.unsqueeze(1) + future_rewards

            td_error = (y - q_s_local).to('cpu').numpy().reshape(-1)

            if self.debug:
                print(f"y = r + 𝛾 * max_a V(s')")
                print(f'r: {rewards}')
                print(f'Actions: {actions}')
                print(f"Q(s' target): {q_s_prime_target}")
                print(f"Q(s' local): {q_s_prime_local}")
                print(f"A(s' local): {a_s_prime_local}")
                print(f"max_a V(s' target): {v_s_prime_target}")
                print(f"future_rewards: gamma * max_a V(s'):\n{future_rewards}")
                print(f"rewards: {rewards}")
                print(f"y = r + future: {y}")
                print(f'q_s_local: {q_s_local}')


        self.optimizer.zero_grad()

        # loss function should only compute (x - y) ** 2 and not mean it
        # loss = F.mse_loss(y, q_s_local) 
        if torch_latest:
            loss_fn = nn.MSELoss(reduction='none')
        else:
            loss_fn = nn.MSELoss(False, False)

        loss = loss_fn(q_s_local, y) * weights
        loss = torch.mean(loss)
        loss.backward()

        self.optimizer.step()

        assert td_error.shape == indices_np.shape
        self.replay_buffer.update_priorities(indices_np, td_error) 

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename = 'qnetwork.pt'):
        torch.save(self.qlocal_network.state_dict(), filename)

    def load(self, filename = 'qnetwork.pt'):
        weights = torch.load(filename)
        self.qlocal_network.load_state_dict(weights)
        self._soft_update(self.qlocal_network, self.qtarget_network, 1.)
        self.eval()

    def eval(self):
        self.qlocal_network.eval()
        self.is_training = False

    def train(self):
        self.qlocal_network.train()
        self.is_training = True