import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import csv
import logging
import sys
import csv
import random
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger()

from pettingzoo.mpe import simple_reference_v3,simple_tag_v3, simple_adversary_v3

def preprocess_state(observation):
    return np.array(observation).flatten()


def preprocess_states(states, num_adversaries=3, state_size_adversary=16, state_size_normal=14, normal_index=3):
    total_length = (num_adversaries * state_size_adversary) + state_size_normal  # This should sum up to 62

    processed_states = np.zeros((len(states), total_length))

    for i, timestep in enumerate(states):
        current_states = []

        for idx, state in enumerate(timestep):
            if idx == normal_index:
                current_states.append(state)
            else:
                current_states.append(state)
        processed_states[i] = np.concatenate(current_states)

    return processed_states #[64,62]


def pad_sequences(sequences, maxlen=None, padding='post', value=0):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    padded_sequences = np.full((len(sequences), maxlen, feature_dim), fill_value=value, dtype=np.float32)

    for i, seq in enumerate(sequences):
        sequence_length = len(seq)
        reshaped_seq = seq.reshape(-1, feature_dim)

        if padding == 'post':
            padded_sequences[i, :sequence_length] = reshaped_seq
        elif padding == 'pre':
            padded_sequences[i, -sequence_length:] = reshaped_seq

    return padded_sequences


class Actor(nn.Module):
    def __init__(self, state_size=14, action_size=5):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)

class Adversary_Actor(nn.Module):
    def __init__(self, state_size=16, action_size=5):
        super(Adversary_Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size_total, action_size_total):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size_total + action_size_total, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output a single Q-value
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)  # Concatenate states and actions
        return self.network(x)


class MADDPGAgent:
    def __init__(self, state_size_normal=14, state_size_adversary=16, action_size=5, num_adversaries=3, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.998, strategy="coop"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.strategy = strategy
        self.num_adversaries = num_adversaries
        self.normal_actor = Actor(state_size=state_size_normal, action_size=action_size).to(device)
        self.adversary_actor = Adversary_Actor(state_size=state_size_adversary, action_size=action_size).to(device)

        total_state_size = (state_size_adversary * num_adversaries) + state_size_normal
        total_action_size = action_size * (num_adversaries + 1)  # One action per agent
        self.critic = Critic(state_size_total=total_state_size, action_size_total=total_action_size).to(device)

        self.actor_optimizer = torch.optim.Adam(
            list(self.normal_actor.parameters()) + list(self.adversary_actor.parameters()), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)


        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


    def act(self, state, adversary):
        if isinstance(state, list):
            state = state[0]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if adversary:
                action = self.adversary_actor(state).cpu().numpy()[0]
            else:
                action = self.normal_actor(state).cpu().numpy()[0]
        return action


    def update(self, experiences, gamma=0.95, tau=0.01):
        states, actions, rewards, next_states, dones = experiences
        if states is None:
            return

        # Convert data to tensors
        states = preprocess_states(states,normal_index=3)
        next_states = preprocess_states(next_states,normal_index=2)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        rewards = adjust_rewards(actions, rewards, strategy=self.strategy)

        # Get next actions from target actors
        next_actions = []
        for i in range(4):
            if i==0 or i==1:
                slice_start = i * 16
                slice_end = slice_start + 16
                actor_actions = self.adversary_actor(next_states[:, slice_start:slice_end])
                next_actions.append(actor_actions)
            if i==2:
                next_actions.append(self.normal_actor(next_states[:, 32:46]))  # Slice for normal actor
            if i==3:
                actor_actions = self.adversary_actor(next_states[:, -16:])
                next_actions.append(actor_actions)
        next_actions = torch.cat(next_actions, dim=1)
        next_Q_values = self.critic(next_states, next_actions.detach())
        rewards = rewards.mean(axis=1, keepdims=True)
        dones = dones.any(axis=1, keepdim=True).float()
        #print(rewards.shape)
        Q_targets = rewards + (gamma * next_Q_values * (1 - dones))
        #print(Q_targets.shape)
        actions = actions.reshape((64,20))
        Q_expected = self.critic(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = torch.cat([self.adversary_actor(states[:,0:16]),self.adversary_actor(states[:,16:32]),
                                  self.adversary_actor(states[:,32:48]),self.normal_actor(states[:,-14:])], dim=1)

        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



def log_to_csv(file_name, episode, total_reward):
    with open(file_name, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward["adversary_0"], total_reward["adversary_1"],
                         total_reward["adversary_2"], total_reward["agent_0"]])

def train_agents(env, num_episodes, agents, batch_size, gamma=0.95, tau=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experience_buffer = deque(maxlen=10000)
    for episode in range(num_episodes):
        env.reset()
        #obs,_,_,_,_ = env.last()
        done = False
        cumulative_rewards = {agent_name: 0 for agent_name in env.agents}

        while not done:
            actions = {}
            for agent in env.agent_iter():
                if agent == "adversary_0":
                    obs,acts,rewards,next_obs,dones = [],[],[],[],[]
                observation, reward, termination, truncation, info = env.last()
                observation = torch.FloatTensor(observation).to(device)
                if termination or truncation:
                    action = None
                else:
                    is_adversary = 'adversary' in agent
                    action = agents.act(observation, adversary=is_adversary)
                actions[agent] = action
                env.step(action)
                if not done:  # Collect experiences if not done
                    next_observation, next_reward, next_term, next_trun, _ = env.last()
                    next_done = next_term or next_trun
                    obs.append(observation.cpu().numpy())
                    next_obs.append(next_observation)
                    acts.append(action)
                    dones.append(next_done)
                    rewards.append(reward)
                    if agent == "agent_0":
                        experience_buffer.append([obs, acts, rewards, next_obs, dones])
                        rewards = torch.FloatTensor(rewards)
                        rewards = adjust_rewards(acts, rewards, strategy=strategy)
                        for i,reward in enumerate(rewards):
                            cumulative_rewards[env.agents[i]] += reward.detach().cpu().item()

                done = termination or truncation

            # Sample a random batch of experiences from the buffer for training
            if len(experience_buffer) > batch_size:
                experiences = random.sample(experience_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                #states = pad_sequences(states)
                #actions = pad_sequences(actions)
                #next_states = pad_sequences(next_states)
                agents.update((states, actions, rewards, next_states, dones), gamma=gamma)


            agents.epsilon = max(agents.epsilon * agents.epsilon_decay, agents.epsilon_min)



        logger.info(f'Episode {episode + 1}, Total Reward: {cumulative_rewards}')
        log_to_csv(file_name,episode + 1, cumulative_rewards)

def adjust_rewards(actions, rewards, strategy="coop"):
    actions = torch.FloatTensor(actions)
    if strategy == "coop":
        cooperative_bonus = torch.mean(actions, dim=-1)
        rewards += cooperative_bonus
    elif strategy == "comp":
        actions_mean = actions.mean(dim=-1, keepdim=True)
        competitive_bonus = torch.abs(actions - actions_mean).sum(dim=-1)
        rewards += competitive_bonus
    return rewards



state_size = 16
action_size = 5
num_episodes = 10000
batch_size = 64
file_name = "tag_MADDPG_comp.csv"
num_agents = 4
strategy = "comp"
env = simple_tag_v3.env(continuous_actions=True,render_mode="ansi",max_cycles=25) #"human" "ansi"
env.reset(seed=42)

agents = MADDPGAgent(strategy=strategy,state_size_normal=14, state_size_adversary=16, action_size=5, num_adversaries=3)
train_agents(env, num_episodes, agents, batch_size)
