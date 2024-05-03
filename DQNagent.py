import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging
import sys
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger()

from pettingzoo.mpe import simple_reference_v3, simple_tag_v3, simple_adversary_v3

def preprocess_state(observation):
    return np.array(observation).flatten()

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.998, hidden_size=50):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.gamma = discount_factor  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        self.model = self._build_model()


    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            #nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            #nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        model.apply(self.init_weights)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def remember(self, state, action, reward, next_state, done):
        state = preprocess_state(state)
        next_state = preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            assert 0 <= action <= 49
            #action = action % 5
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(target_f, target_f.detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def log_to_csv(file_name, episode, total_reward):
    with open(file_name, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward["agent_0"],total_reward["agent_1"]])


def train(env, num_episodes, batch_size, agent1, agent2):
    for episode in range(num_episodes):
        env.reset()
        cumulative_rewards = {agent_name: 0 for agent_name in env.agents}
        done = False

        while not done:
            for agent_name in env.agent_iter():

                observation, reward, termination, truncation, info = env.last()
                done = termination or truncation

                if done:
                    action = None
                else:
                    if "agent_0" in agent_name:
                        if random.random() < agent1.epsilon:
                            action = env.action_space("agent_0").sample()
                            #print(f"random:{action}")
                        else:
                            action = agent1.act(preprocess_state(observation))
                            #print(f"act:{action}")
                        env.step(action)
                        next_observation, _, _, _, _ = env.last()
                        agent1.remember(observation, action, reward, next_observation, done)
                        agent1.replay(batch_size)
                    elif "agent_1" in agent_name:
                        if random.random() < agent2.epsilon:
                            action = env.action_space("agent_1").sample()
                        else:
                            action = agent2.act(preprocess_state(observation))
                        env.step(action)
                        next_observation, _, _, _, _ = env.last()
                        agent2.remember(observation, action, reward, next_observation, done)
                        agent2.replay(batch_size)

                cumulative_rewards[agent_name] += reward

                if done:
                    break

        logger.info(f'Episode {episode + 1}, Total Reward: {cumulative_rewards}')
        log_to_csv(file_name,episode + 1, cumulative_rewards)


state_size = 21
action_size = 50
num_episodes = 10000
batch_size = 64
file_name = "DQN_1.csv"

env = simple_reference_v3.env(local_ratio=1,continuous_actions=False,render_mode="ansi") #"human" "ansi"
env.reset(seed=42)
agent1 = DQNAgent(state_size, action_size)
agent2 = DQNAgent(state_size, action_size)

train(env, num_episodes, batch_size, agent1, agent2)


