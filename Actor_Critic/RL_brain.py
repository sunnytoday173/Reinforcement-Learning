import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

np.random.seed(2)
torch.manual_seed(2)
HIDDEN_DIM=20

class ActorNet(nn.Module):
    def __init__(self,n_features,n_actions):
        super(ActorNet,self).__init__()
        self.fc1=nn.Linear(n_features,HIDDEN_DIM)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(HIDDEN_DIM,n_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        actions_value = F.softmax(self.out(x))
        return actions_value

class CriticNet(nn.Module):
    def __init__(self,n_features):
        super(CriticNet,self).__init__()
        self.fc1 = nn.Linear(n_features,HIDDEN_DIM)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(HIDDEN_DIM,1)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        value = self.out(x)
        return value

class Actor(object):
    def __init__(self,n_features,n_actions,learning_rate=0.001):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.actor_net = ActorNet(n_features, n_actions)
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=learning_rate)

    def choose_action(self,x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        actions_prob = self.actor_net.forward(x)
        actions_prob = actions_prob.data.numpy()
        action  = np.random.choice(range(actions_prob.shape[1]),p=actions_prob.ravel())
        return action

    def learn(self,s,a,td_error):
        td_error=td_error.detach()
        s = Variable(torch.FloatTensor(s[np.newaxis,:]))
        actions_prob = self.actor_net(s)
        log_prob = torch.log(actions_prob[0,a])
        loss = -torch.mean(log_prob * td_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

class Critic(object):
    def __init__(self,n_features,reward_decay=0.9,learning_rate=0.01):
        self.n_features=n_features
        self.gamma = reward_decay
        self.lr=learning_rate
        self.critic_net=CriticNet(n_features)
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=learning_rate)

    def learn(self,s,r,s_):
        s = Variable(torch.FloatTensor(s[np.newaxis,:]))
        s_ = Variable(torch.FloatTensor(s_[np.newaxis,:]))
        v_next = self.critic_net(s_).detach()
        v_eval =  self.critic_net(s)
        v_target = r+self.gamma*v_next
        td_error = v_target-v_eval
        loss = td_error**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error


