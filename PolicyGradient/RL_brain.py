import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)
HIDDEN_DIM=50
#Father Net
class Net(nn.Module):
    def __init__(self,n_features,n_actions):
        super(Net,self).__init__()
        self.fc1=nn.Linear(n_features,HIDDEN_DIM)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(HIDDEN_DIM,n_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.tanh(self.fc1(x))
        actions_value = F.softmax(self.out(x))
        return actions_value

#Policy Gradient
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.96,
    ):
        self.n_actions = n_actions
        self.n_features =n_features
        self.lr = learning_rate
        self.gamma= reward_decay
        self.policy_net = Net(n_features,n_actions)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self,x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        actions_prob = self.policy_net.forward(x)
        actions_prob = actions_prob.data.numpy()
        action  = np.random.choice(range(actions_prob.shape[1]),p=actions_prob.ravel())
        return action

    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm =  Variable(torch.FloatTensor(self._discount_and_norm_rewards()))
        b_s = Variable(torch.FloatTensor(np.vstack(self.ep_obs)))
        b_a = Variable(torch.LongTensor(np.array(self.ep_as).astype(np.int64)))
        all_actions=self.policy_net(b_s)
        neg_log_prob = Variable(torch.zeros(len(all_actions)))
        for i in range(len(neg_log_prob)):
            neg_log_prob[i] = self.loss_func(all_actions[i,:].view(1,-1),b_a[i])
        loss = torch.mean(neg_log_prob*discounted_ep_rs_norm)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return discounted_ep_rs_norm.data.numpy()

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs