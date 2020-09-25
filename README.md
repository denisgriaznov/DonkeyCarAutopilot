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

The model used here only predict the steering value, keeping the throttle constant. But on a mountain track, speed does not only depend on the throttle, and there are moments when the car is stationary. In addition to being pointless, it also has a negative impact on training, as the agent is rewarded all the time.

The obvious solution is to set the PID controller to speed, which I did. I also noticed that the speed when moving backward returns without minus, so I limited the pid to zero.

Next, let's look at the structure of the neural network. In the input, it recieves a stack of 4 consecutive frames, first converting them to gray. Since the main landmark on the road is the central yellow line. We can assume that it stands out weak in gray mode, we can try to apply some kind of filter. But first, I just change it to color images resized to 120 * 120. The input layer has the dimensions [120, 120, 12]

The output is for categorical selection. This is logical, but first you can reduce the number of classes from 15 to 9.

