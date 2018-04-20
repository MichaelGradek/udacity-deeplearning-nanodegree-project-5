import sys
from agents.agent import Agent
from task import Task
import numpy as np

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {l : [] for l in labels}

num_episodes = 500
target_pos = np.array([0., 0., 10.])
task = Task(init_pose=target_pos, target_pos=target_pos)
agent = Agent(task)
rewards = []

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    ave_reward = 0
    cnt = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        ave_reward += reward
        cnt += 1
        if i_episode == 500:
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
        if done:
            ave_reward /= cnt
            print("\rEpisode = {:4d}, score = {:7.3f} (reward = {:7.3f})".format(i_episode, agent.score, ave_reward), end="")  # [debug]
            break
    rewards.append(ave_reward)
    sys.stdout.flush()
