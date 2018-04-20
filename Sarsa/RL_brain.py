import numpy as np
import pandas as pd

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

