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
    def __init__(self,n_states,n_actions):
        super(Net,self).__init__()
        self.fc1=nn.Linear(n_states,HIDDEN_DIM)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(HIDDEN_DIM,n_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


#Deep Q Netword off-policy
class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_states,
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            epsilon_greedy_increment=None
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_greedy_increment
        self.epsilon = 0 if epsilon_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter=0
        self.memory_counter=0
        self.memory = np.zeros((self.memory_size,n_states*2+2))
        self.eval_net,self.target_net=Net(n_states,n_actions),Net(n_states,n_actions)
        self.optimizer=optim.Adam(self.eval_net.parameters(),lr=learning_rate)
        self.loss_func=nn.MSELoss()
        self.cost_his = []

    def choose_action(self,x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        # input only one sample
        if np.random.uniform()<self.epsilon:
            actions_value=self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()
            action = int(action[0])
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index,:]=transition
        self.memory_counter +=1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.replace_target_iter ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #sample batch transitions
        sample_index = np.random.choice(self.memory_size,self.batch_size)
        b_memory = self.memory[sample_index,:]
        b_s = Variable(torch.FloatTensor(b_memory[:,:self.n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:,self.n_states:self.n_states+1].astype(np.int64)))
        b_r = Variable(torch.FloatTensor(b_memory[:,self.n_states+1:self.n_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:,self.n_states+2:]))
        # q_eval with regard to the action in experience
        q_eval =self.eval_net(b_s).gather(1,b_a) # shape(batch,1)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma*torch.max(q_next,1)[0].view(self.batch_size,1)
        loss = self.loss_func(q_eval,q_target)
        #print(loss.data.numpy()[0])
        self.cost_his.append(loss.data.numpy()[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max



    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


class RL(object):
    def __init__(self,actions,learning_rate=0.01,reward_dacay=0.9,epsilon_greedy=0.9):
        self.actions=actions
        self.lr=learning_rate
        self.gamma=reward_dacay
        self.epsilon=epsilon_greedy
        self.q_table=pd.DataFrame(columns=self.actions)

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self,observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self,*args):
        pass

#off-policy
class QLearningTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_dacay=0.9,epsilon_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_dacay, epsilon_greedy)

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


#on-policy
class SarsaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_dacay=0.9,epsilon_greedy=0.9):
        super(SarsaTable,self).__init__(actions,learning_rate,reward_dacay,epsilon_greedy)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s,a]
        if s_!='terminal':
            q_target= r +self.gamma*self.q_table.ix[s_,a_]
        else:
            q_target=r
        self.q_table.ix[s,a]+=self.lr*(q_target-q_predict)

class SarsalambdaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_dacay=0.9,epsilon_greedy=0.9,trace_decay=0.9):
        super(SarsalambdaTable,self).__init__(actions,learning_rate,reward_dacay,epsilon_greedy)
        self.lambda_=trace_decay
        self.eligibility_trace = self.q_table.copy()


    def check_state_exist(self,state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s,a]
        if s_!='terminal':
            q_target= r +self.gamma*self.q_table.ix[s_,a_]
        else:
            q_target=r
        error=q_target-q_predict
        #Method 1
        #self.eligibility.trace.ix[s,a]+=1
        #Method 2
        self.eligibility_trace.ix[s,:]*=0
        self.eligibility_trace.ix[s,a]=1
        self.q_table+=self.lr*error*self.eligibility_trace
        self.eligibility_trace*=self.gamma*self.lambda_

