import os
import random

import numpy as np
import gym
import cv2
import gym_donkeycar
from simple_pid import PID

from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras import backend as K



EPISODES = 10000

# Max square resolution is 120
img_rows, img_cols = 120, 120

# Use remote or change path
SIM_PATH = "C:\\Users\\Denis\\Desktop\\GymCarControl\\DonkeySimWin_Last\\donkey_sim.exe"
#SIM_PATH = "remote"

PORT = 9091
ENV_NAME = 'donkey-mountain-track-v0'
IS_TRAIN = False

#Set speed setpoint for PID
THROTTLE = 0.3
SPEED_SETPOINT = 5.0

MODEL_NAME = "rl_driver.h5"

# Customized bins count
CTGR_COUNT = 9

img_channels = 4 

class DQNAgent:

    def __init__(self, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train
        self.action_space = action_space
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()


    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #120*120*4
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 9 categorical bins for steering angles
        model.add(Dense(CTGR_COUNT, activation="linear"))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model


    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def process_image(self, obs):
        obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
        return obs


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    # Over time, the balance shifts from exploitation to exploitation
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()[0]
        else:
            q_value = self.model.predict(s_t)
            return linear_unbin(q_value[0])


    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)


    def load_model(self, name):
        self.model.load_weights(name)


    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

# bin and unbin functions in general view

def linear_bin(a):
    a = a + 1
    b = round(a / (2 / (CTGR_COUNT-1)))
    arr = np.zeros(CTGR_COUNT)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    if not len(arr) == CTGR_COUNT:
        raise ValueError('Illegal array length, must be {}'.format(CTGR_COUNT))
    b = np.argmax(arr)
    a = b * (2 / (CTGR_COUNT-1)) - 1
    return a


def run_ddqn():
    # only needed if TF==1.13.1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    conf = {"exe_path" : SIM_PATH,
        "host" : "127.0.0.1",
        "port" : PORT,
        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "Denis",
        "font_size" : 100,
        }


    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(ENV_NAME, conf=conf)

    action_space = env.action_space # Steering and Throttle


    try:
        agent = DQNAgent(action_space, train=IS_TRAIN)
        episodes = []

        if os.path.exists(MODEL_NAME):
            print("load the saved model")
            agent.load_model(MODEL_NAME)

        # Initialize PID for throttle
        pid = PID(1, 0.1, 0.1, setpoint=SPEED_SETPOINT)
        pid.output_limits = (0, 1)

        for e in range(EPISODES):

            #if agent.epsilon <= agent.epsilon_min:
            #     break
            print("Episode: ", e)
            throttle = THROTTLE # Set throttle as constant value

            done = False
            obs = env.reset()

            episode_len = 0

            x_t = agent.process_image(obs)

            s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4

            while not done:

                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)
                #cv2.imwrite('color_img.jpg',next_obs)
                throttle = pid(info.get('speed'))

                x_t1 = agent.process_image(next_obs)

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #1x80x80x4
                agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    agent.train_replay()

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 30 == 0:
                    print('SPEED: {} , CTE: {} , HIT: {}'.format(info.get('speed'),info.get('cte'),info.get('hit')))
                    #print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , agent.max_Q)

                if done:
                    # Every episode update the target model to be same with model
                    agent.update_target_model()
                    episodes.append(e)
                    # Save model for each episode
                    if agent.train:
                        agent.save_model(MODEL_NAME)

                    print("episode:", e, "  memory length:", len(agent.memory),
                        "  epsilon:", agent.epsilon, " episode length:", episode_len)

    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()


if __name__ == "__main__":

    run_ddqn()
