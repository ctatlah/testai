'''
Created on Apr 3, 2024

@author: ctatlah
'''
#
# imports
#

import time
import gym
import PIL.Image

import numpy as np
import tensorflow as tf
import com.test.ai.utils.landerUtils as utils
from com.test.ai.data.DataLoader import LoadData

from collections import deque, namedtuple
from pyvirtualdisplay import Display
from tensorflow.keras import Sequential #@UnresolvedImport
from tensorflow.keras.layers import Dense, Input #@UnresolvedImport
from tensorflow.keras.losses import MSE #@UnresolvedImport
from tensorflow.keras.optimizers import Adam #@UnresolvedImport

#
# setup
#
Display(visible=0, size=(840, 480)).start();
tf.random.set_seed(0)

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

# env setup
env = gym.make('LunarLander-v2')
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
state_size = env.observation_space.shape
num_actions = env.action_space.n
print('State Shape:', state_size)
print('Number of actions:', num_actions)


#
# Work
#
print ('Here we go, using reinforcement learning to land a lunar lander')
current_state = env.reset() # Reset the environment and get the initial state.

action = 0 #select action
next_state, reward, done, _ = env.step(action) # Run a single time step of the environment's dynamics with the given action.
utils.display_table(current_state, action, next_state, reward, done) # Display table with values.
current_state = next_state # Replace the `current_state` with the state after the action is taken

# Q-Network
q_network = Sequential([
    Input(shape=state_size),                      
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'),
    ])
# Target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),                       
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'),
    ])
optimizer = Adam(learning_rate=ALPHA) 

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


# Train the Agent
#
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            utils.agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)
        
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
utils.plot_history(total_point_history)

import logging
logging.getLogger().setLevel(logging.ERROR)
filename = LoadData().resFolder / "lunar_lander.mp4"
utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
