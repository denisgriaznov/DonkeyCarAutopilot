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
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras import backend as K

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy


EPISODES = 10000
img_rows, img_cols = 80, 80
#SIM_PATH = "C:\\Users\\Denis\\Desktop\\GymCarControl\\DonkeySimWin_Last\\donkey_sim.exe"
SIM_PATH = "remote"
PORT = 9091
ENV_NAME = 'donkey-mountain-track-v0'
IS_TRAIN = False
THROTTLE = 0.3
SPEED_SETPOINT = 5.0
MODEL_NAME = "rl_driver.h5"
CTGR_COUNT = 9
# Convert image into Black and white
img_channels = 3 # We stack 4 frames

def build_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    # 15 categorical bins for Steering angles
    model.add(Dense(2, activation="linear"))

    return model



if __name__ == "__main__":
    model = build_model()
    conf = {"exe_path" : SIM_PATH,
        "host" : "127.0.0.1",
        "port" : PORT,
        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "Denis",
        }
    env = gym.make(ENV_NAME, conf=conf)

    action_space = env.action_space # Steering and Throttle

    policy = EpsGreedyQPolicy()
    sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)
    sarsa.compile('adam', metrics = ['mse'])
    sarsa.fit(env, nb_steps = 5000, visualize = False, verbose = 1)
    run_ddqn()
