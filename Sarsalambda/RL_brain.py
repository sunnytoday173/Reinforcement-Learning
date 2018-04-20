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

