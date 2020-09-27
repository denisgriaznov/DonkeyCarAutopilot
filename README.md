# DonceyCarAutopilot

Implementing an autopilot for a car in a gym environment. Based on the following sources:

https://gym.openai.com/

https://github.com/tawnkramer/gym-donkeycar

https://docs.donkeycar.com/

## Review

The gym-donkeycar library is an adaptation of the Donkeycar environment in Unity for reinforcement learning using the gym library.

The RGB image from car camera is returned as the environment state. This is an array [120,160,3]. The reward is also returned, depending on the distance from the middle of the road. 

Based on this data, the reinforcement learning agent must rebuild a model that predicts two variables: steering rotation and throttle value.

This project uses the track 'donkey-mountain-track-v0'. Unlike the others, it has height differences, which is reflected in learning.

Examples from the repository are taken as a basis:

https://github.com/tawnkramer/gym-donkeycar/tree/master/examples/reinforcement_learning

## DDQN

Since the input of the model is an image, it is logical to use a convolutional neural network as a model.
I built the model like this:

#### Input(120,120,4) -> Conv2d(64) -> Conv2d(64) -> Conv2d(128) -> Conv2d(128) -> Dense(512) -> Output(9)

## PPO

## Tests

## Resume
