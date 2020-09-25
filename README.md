# DonkeyCarAutopilot
A project to create automatic control based on reinforcement learning.
The project used the following resources:

https://github.com/tawnkramer/gym-donkeycar

https://gym.openai.com/

## Review

Using of reinforcement learning is justified by the nature of the vehicle driving problem, 
as well as by the gym environment, which is adapted specifically for such models.

In this case, the reinforcement learning model can be interpreted as follows:

- The agent sends an action to the environment - rudder position and speed. It is an array of two numbers from -1 to 1.

- In response, the environment sends the agent the state after this action - rgb image from the car camera as an array [120, 160, 3]. The agent also receives a reward - a value, depending on which the weights of the predictive model change.

The need for image processing drives to using convolutional neural network. 
We see the use of reinforcement learning and convolutional neural networks in this example:

https://github.com/tawnkramer/gym-donkeycar/blob/master/examples/reinforcement_learning/ddqn.py

I took it as a basis.

## Model tuning

