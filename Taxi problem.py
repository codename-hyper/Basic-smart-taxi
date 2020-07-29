import numpy as np
import gym

env = gym.make("Taxi-v3")

# state = env.reset()

n_states = env.observation_space.n
n_actions = env.action_space.n

# <editor-fold desc="Without Q (uncomment inside to run)">
# # implementing loop in RL
# state = env.reset()
# counter = 0
# g = 0
# reward = None
#
# # while loop
# while reward != 20:
#     state, reward, done, info = env.step(env.action_space.sample())
#     counter += 1
#     g += reward
#     # env.render()
#
# # show repeated times
# print("Solved in {} Steps with a total reward of {}".format(counter, g))
# </editor-fold>

# <editor-fold desc="With Q">

# initiate Q learning Matrix
Q = np.zeros([n_states, n_actions])

# <editor-fold desc="loop to run 1 episode">
episodes = 1  # episode is timestep
G = 0  # G is Total of rewards
alpha = 0.618  # alpha is learning parameter

for episode in range(1, episodes + 1):
    done = False
    G, reward = 0, 0
    state = env.reset()
    firstState = state
    print("Initial State = {}".format(state))
    while reward != 20:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
        G += reward
        state = state2

finalState = state
# </editor-fold>

# <editor-fold desc="loop to run 2000 episode">

episodes = 2000
rewardTracker = []
G = 0
alpha = 0.618

for episode in range(1, episodes + 1):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state, action] += alpha * ((reward + (np.max(Q[state2])) - Q[state, action]))
        G += reward
        state = state2

    if episode % 100 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))

# </editor-fold>

# <editor-fold desc="Final run with optimal Q">

state = env.reset()
done = None

while done != True:
    # We simply take the action with the highest Q Value
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    env.render()

# </editor-fold>

# </editor-fold>
