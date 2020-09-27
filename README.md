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

4 consecutive frames in gray mode are fed to the input. The output represents 9 categories of different steering positions. The throttle is supposed to be kept constant. But on a mountain track, this leads to a stop:

(IMAGE STOPP)

Firstly, this is bad for itself, and secondly, the agent is charged a large reward, since the car does not crash anywhere.

To fix this I set the PID controller to the speed value from the info.

    pid = PID(1, 0.1, 0.1, setpoint=SPEED_SETPOINT)
    pid.output_limits = (0, 1)
    ....
    throttle = pid(info.get('speed'))


## PPO

## Tests

## Resume
