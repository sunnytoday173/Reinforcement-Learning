import numpy as np
import pandas as pd
import time

np.random.seed(2) # for reproduce

N_STATES = 6 # length
ACTIONS = ['left','right'] #actions
EPSILON = 0.9 # epsilon-greedy
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
MAX_EPOCHS = 13 # maximum epochs
FRESH_TIME =0.1 # fresh time for one move

def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))), # initialize
        columns = actions, # actions' name
    )
    print(table)
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform()>EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(state,action):
    if action == 'right':
        if state == N_STATES - 2:
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state+1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_ = state
        else:
            state_ = state - 1
    return state_,reward

def update_env(state,epoch,step_counter):
    env_list = ['-']*(N_STATES-1)+['T']
    if state == 'terminal':
        interaction = 'Epoch %s: total_steps = %s' %(epoch+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r')
    else:
        env_list[state]='o'
        interaction=''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES,ACTIONS)
    for epoch in range(MAX_EPOCHS):
        step_encounter=0
        state = 0
        is_terminated = False
        update_env(state,epoch,step_encounter)
        while not is_terminated:
            action = choose_action(state,q_table)
            state_,reward = get_env_feedback(state,action)
            q_predict = q_table.ix[state,action]
            if state_ != 'terminal':
                q_target = reward+GAMMA*q_table.iloc[state_,:].max()
            else:
                q_target = reward
                is_terminated = True
            q_table.ix[state,action]+= ALPHA *(q_target-q_predict)
            state = state_
            update_env(state,epoch,step_encounter+1)
            step_encounter+=1
    return q_table

if __name__ == '__main__':
    q_table=rl()
    print('\r\nQ-table:')
    print(q_table)
#build_q_table(N_STATES,ACTIONS)