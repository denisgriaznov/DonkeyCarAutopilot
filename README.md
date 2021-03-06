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

![stop](/images/zerospeed.gif)

Firstly, this is bad for itself, and secondly, the agent is charged a large reward, since the car does not crash anywhere.

To fix this I set the PID controller to the speed value from the info.

    pid = PID(1, 0.1, 0.1, setpoint=SPEED_SETPOINT)
    pid.output_limits = (0, 1)
    ....
    throttle = pid(info.get('speed'))

This way the speed is kept constant. 

Each step is either exploration (random steering position) or exploitation (predicting the current model based on the image). For each step, the model gives out the steering position and then receives a reward. When the episode ends, the model's weights change depending on the amount of the reward. Over time, the probability of exploitation (epsilon) decreases linearly.

After training the agent for 10,000 episodes, unsatisfactory result was obtained:

![ddqn](/images/testddqn.gif)

Possible ways to improve:

- replace gray with selection of lines, for example Canny
- limit the steering wheel turn to values -0.5 and 0.5
- change the policy to a more flexible one (non-linear decrease epsilon)

Alternatively, you can try supervised learning. I created a script that control both the speed and distance from the middle of the road using pid, and then collects a dataset from these values and the corresponding images:

https://github.com/denisgriaznov/DonkeyCarAutopilot/blob/master/code/data_collect.py

You can try to train this to predict the distance to the center or the desired angle of rotation.



## PPO


Since the first example used a very simple policy of the relationship between exploitation and exploitation (epsilon gradually decreased and represented the probability of an exploitation step), let's try using PPO. This is a more adaptive algorithm that will smooth out the change in balance at each step.

https://openai.com/blog/openai-baselines-ppo/

Here I also used a convolutional neural network to process state, however out of the box library stable-baselines.

    model = PPO2(CnnPolicy, env, verbose=1)

The model was trained for 10,000 steps (about 80 episodes)
Result:

![ppo](/images/testppo.gif)

Then I started two cars at the same time as a simulated competition. And two models were trained at 50,000 steps.
Result:

![ppocompetition](/images/testcompetition.gif)

Possible ways to improve:

- control the speed, as the cars go very slowly
- try the CNN-LSTM policy as the solution may depend on the sequence of frames

## Tests

Here I provide test results for single movement using PPO2

Distribution of time for one step im ms:

![time](/images/timedist.png)

Distribution of reward summ for 50 episodes:

![reward](/images/rewarddist.png)

Distribution of episode length for 50 episodes:

![len](/images/lendist.png)


Unfortunately, even long duration and rewards do not indicate success, as they can be obtained by "cheating".

## Resume

The main problem for this unsupervised learning model is how the reward is calculated. So, the agent will receive a reward if he drives around him or stands still. It would be nice to revise the procedure for receiving rewards depending on:

- speed
- driving directions
- traversed path in the right direction

I also ran into some minor technical difficulties. For example, gym-donkeycar does not return a negative value when moving backward, as they say in their documentation.
Also, the mountain track is present in the only (not the last) release of donkeycar:

https://github.com/tawnkramer/gym-donkeycar/releases/tag/v2020.5.16
