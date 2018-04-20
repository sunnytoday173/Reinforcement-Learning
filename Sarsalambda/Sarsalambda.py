from maze_env import Maze
from RL_brain import SarsalambdaTable
def update():
    for epoch in range(100):
        #intial observation
        observation = env.reset()
        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        while True:
            # fresh env
            env.render()
            #RL take action and get next observation and reward
            observation_,reward,done = env.step(action)
            #RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            #RL learn from this transition(s,a,r,s,a) -> Sarsa
            RL.learn(str(observation),action,reward,str(observation_),action_)
            #swap observation and action
            observation = observation_
            action = action_
            # break while looop when end of this epoch
            if done:
                break
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL =SarsalambdaTable(actions=list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()


