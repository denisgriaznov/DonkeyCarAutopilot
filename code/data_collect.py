import os
import random

import numpy as np
import gym
import cv2
import gym_donkeycar
from simple_pid import PID

EPISODES = 10000
SIM_PATH = "remote"
PORT = 9091
ENV_NAME = 'donkey-mountain-track-v0'
CTE_SETPOINT = 0
SPEED_SETPOINT = 5.0

THROTTLE = 0.3
STEERING = 0


def run_ddqn():

    conf = {"exe_path" : SIM_PATH,
        "host" : "127.0.0.1",
        "port" : PORT,
        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "Denis",
        }

    env = gym.make(ENV_NAME, conf=conf)

    action_space = env.action_space # Steering and Throttle


    try:

        episodes = []

        # Initialize PID for throttle and steering
        pidThrottle = PID(1, 0.1, 0.1, setpoint=SPEED_SETPOINT)
        pidThrottle.output_limits = (0, 1)
        #pidSteering = PID(0.5, 0.5, 0.5, setpoint=CTE_SETPOINT)
        pidSteering = PID(1, 0.2, 0.2, setpoint=CTE_SETPOINT)
        pidSteering.output_limits = (-0.5, 0.5)


        for e in range(EPISODES):

            print("Episode: ", e)
            throttle = THROTTLE # Set throttle as constant value
            steering = STEERING

            done = False
            obs = env.reset()

            episode_len = 0
            num = 214
            while not done:

                episode_len = episode_len + 1

                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)
                cv2.imwrite('color_img.jpg',next_obs)
                throttle = pidThrottle(info.get('speed'))
                steering = pidSteering(info.get('cte'))



                if episode_len % 5 == 0:
                    dataset = open('dataset.txt', 'a+')
                    dataset.write("\n" + '{}, {}, {}'.format(num,steering,info.get('cte')))
                    dataset.close()
                    cv2.imwrite('imageset/{}.jpg'.format(str(num)),next_obs)
                    num = num + 1


    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()

if __name__ == "__main__":

    run_ddqn()
