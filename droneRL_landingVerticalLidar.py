import tensorflow as tf
import time
import numpy as np
import random
import airsim
import os
import math
import pickle

BATCH_SIZE = 128
MIN_ALTITUDE = 5
MAX_ALTITUDE = 10
MIN_EPSILON = 0.001
MAX_EPSILON = 0.4
LAMBDA = 0.001
GAMMA = 0.85
OBSERVE = 20
EXE_TIME = 0.1

tf.compat.v1.disable_eager_execution()

def reward(client, height, trigger, actual_episode_length, max_episode_length, new_vel_z):

    reward = 0.0
    #collision_info = client.simGetCollisionInfo() Not reliable for landing

    if height <= 0.2 and new_vel_z > 0.8:
        reward = -10.0
    else:
        if height <= 0.2 and trigger == 1:
            print("## Landed ##")
            reward = 10.0
        elif height > 0.2 and trigger == 1:
            reward = -3.0
        elif height > 1 and height <= 1.5 and new_vel_z > 0.5:
            reward = -0.5
        elif height > 1.8 and height <= 4 and new_vel_z < 0.8:
            reward = -0.5
        elif height > 4 and new_vel_z < 1.8:
            reward = -0.5
        else:
            reward = -0.01

        if max_episode_length < actual_episode_length:
            reward = -10.0

    return reward

def interpret_action(action):
    if action == 0:
        quad_offset = (0, 0, 1, 0)
    elif action == 1:
        quad_offset = (0, 0, 0.3, 0)
    elif action == 2:
        quad_offset = (0, 0, 2, 0)
    elif action == 3:
        quad_offset = (0, 0, 0, 1)
    return quad_offset

def go_back(client, home):
    client.moveToPositionAsync(home.x_val, home.y_val, -1*np.random.randint(MIN_ALTITUDE,MAX_ALTITUDE+1), 5).join()
    time.sleep(0.5)
    client.moveByVelocityAsync(0, 0, -0.0, 5).join()

def distance_2d(pos, home):
    return np.sqrt((home.x_val-pos[0])**2 + (home-z_val-pos[1])**2)

def is_done(client, pos, trigger, actual_episode_lenght, max_episode_lenght, new_vel_z):
    height = np.abs(pos[2])
    return client.simGetCollisionInfo().has_collided or (height <= 0.2 and new_vel_z > 0.8) or (height <= 0.2 and trigger == 1) or (height > 0.2 and trigger == 1) or max_episode_lenght < actual_episode_lenght

def store_transition(mem,store_or_read):
    store_path = 'replay_experiencesLidar.pkl'
    if(store_or_read=='read'):
        if not os.path.exists(store_path) or os.path.getsize(store_path)==0:
            print('Not Found the pkl file!')
            return mem, False
        else:
            store_file = open(store_path,'rb')
            #mem.set_samples(pickle.load(store_file))
            mem = pickle.load(store_file)
            store_file.close()
            memory_len = mem.get_len()
            print('Successfully load the replay_experiencesLidar.pkl, %05d memory'%memory_len)
            return mem, True
    elif(store_or_read=='store'):
        store_file = open(store_path, 'wb')
        #pickle.dump(mem.get_samples, store_file)
        pickle.dump(mem, store_file)
        store_file.close()
        return 1
    else:
        return 0

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.compat.v1.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.compat.v1.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.compat.v1.layers.dense(self._states, 64, activation=tf.nn.relu)
        fc2 = tf.compat.v1.layers.dense(fc1, 64, activation=tf.nn.relu)
        fc3 = tf.compat.v1.layers.dense(fc2, 32, activation=tf.nn.relu)
        self._logits = tf.compat.v1.layers.dense(fc2, self._num_actions)
        loss = tf.compat.v1.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.compat.v1.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    def get_len(self):
        return len(self._samples)

    def get_samples(self):
        return self._samples

    def set_samples(self, samples):
        self._samples = samples

class GameRunner:
    def __init__(self, sess, model, memory, max_eps, min_eps, episode_length):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._eps = self._max_eps
        self._steps = 0
        self.episode_length = episode_length
                   
    def run(self, client, cnt, already_saved):
        
        tot_reward = 0

        # action loop
        for j in range(self.episode_length):
            
            # get current state
            print("===== Action " + str(j))
            curr_state = client.getLidarData().pose.position.z_val
            z_val = np.abs(client.getMultirotorState().kinematics_estimated.position.z_val)
            input_state = np.array([[curr_state]])

            print("  INITIAL STATE " + str(curr_state))

            #Taking action index

            #At least same of the initial experiences are positive
            if self._memory.get_len() <= OBSERVE*(self.episode_length)/4:
                if z_val <= 0.2:
                    action_index = 3
                elif z_val >= 4:
                    action_index = 2
                elif z_val >= 1.8:
                    action_index = 0
                else: 
                    action_index = 1

            elif self._memory.get_len() > OBSERVE*(self.episode_length)/4 and self._memory.get_len() <= OBSERVE*(self.episode_length)/3 and already_saved:
                action_index = np.random.randint(0, self._model._num_actions)
                print('episode=%05d,step=%05d,we are observing the env,the action is random......'%(cnt,j))
            else:
                action_index = self._choose_action(input_state)

            # calculate action input to AirSim
            next_action = interpret_action(action_index)

            new_vel_x = next_action[0]
            new_vel_y = next_action[1]
            new_vel_z = next_action[2]
            trigger = next_action[3]
            print("  ====== moving at (" + str(new_vel_x) + " " + str(new_vel_y) + " " + str(new_vel_z)  + "), trigger ",trigger)
            client.moveByVelocityAsync(new_vel_x, new_vel_y, new_vel_z, 1).join()

            time.sleep(EXE_TIME)

            new_state = client.getLidarData().pose.position.z_val
            reward_current = reward(client, new_state, trigger, j, episode_length, new_vel_z)

            self._memory.add_sample((curr_state, action_index, reward_current, new_state))    

            if(self._memory.get_len()<2500):
                signal_back = store_transition(self._memory, 'store')
                if signal_back:
                    print('######## The replay experience has been saved successfully after %d episodes ##########' % cnt)
                else:
                    print('######## Warning: the replay experiences can not be saved after %d episodes ##########' % cnt)     

            # move the agent to the next state and accumulate the reward
            tot_reward += reward_current

            print("Episode {}, Total reward: {}, Eps: {}".format(cnt, tot_reward, self._eps))

            if is_done(client, current_position, trigger, j, episode_length, new_vel_z):
                print("### Episode ended.")
                break

    def _choose_action(self, state):
        if random.random() < self._eps:
            return np.random.randint(0, self._model._num_actions)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([[val[0] for val in batch]]).T
        next_states = np.array([[(np.zeros(self._model._num_states)
                                          if val[3] is None else val[3]) for val in batch]]).T
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
        # get the current q values for all actions in state
        current_q = q_s_a[i]
        # update the q value for action
        if next_state is None:
            # in this case, the game completed after action, so there is no max Q(s',a')
            # prediction possible
            current_q[action] = reward
        else:
            current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
        x[i] = state
        y[i] = current_q
        self._model.train_batch(self._sess, x, y)


if __name__ == "__main__":

    num_states = 1
    num_actions = 4

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    with tf.compat.v1.Session() as sess:

        num_episodes = 300
        episode_length = 15
                   
        trainables = tf.compat.v1.trainable_variables()
        trainable_saver = tf.compat.v1.train.Saver(trainables)

        mem, already_saved = store_transition(mem,'read')

        sess.run(tf.compat.v1.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(os.path.join("saved_networks","verticalOpt"))
        print('checkpoint:', checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
            already_saved = True
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            if not os.path.exists(os.path.join("saved_networks","verticalOpt")):
                os.mkdir(os.path.join("saved_networks","verticalOpt"))
                print('The file not exists, is created successfully')
            already_saved = False
            print("Could not find old network weights")
        
        gr = GameRunner(sess, model, mem, MAX_EPSILON, MIN_EPSILON, episode_length)

        # init drone
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)

        #taking home position
        home = client.getMultirotorState().kinematics_estimated.position
                   
        cnt = 1
        while cnt < num_episodes:
            go_back(client, home)
            gr.run(client, cnt, already_saved)
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                #Save and train model every 10 episodes
                if gr._memory.get_len() >= OBSERVE*episode_length/3 or already_saved:
                    print("### TRAINING ###")
                    gr._replay()
                    # exponentially decay the eps value
                    gr._steps += 1
                    gr._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                                  * math.exp(-LAMBDA * gr._steps)
                    trainable_saver.save(sess, os.path.join("saved_networks","verticalOpt","Simply_maze"),global_step=cnt,write_state=True)
                    print('######## The mode has been saved successfully after %d episodes ##########' % cnt)
               
            cnt += 1

    
