## Project Description
The environment in this repository consist of 2 paddles and a ball in a table tennis-like environment. The agents controlling each paddle receives a reward of +0.1 every time it hits the ball over the net and is instead penalized with a score of -0.01 if it either let the ball drop or send it out of bounds.

Each agent work in a 8 dimensional observation space consisting of position and velocity of the ball and the paddle. The action space is instead continous and 2 dimensional representing x and y movement of the paddle.

The environment is considered solved with a score of 0.5 consisting of the maximum value registered during an episode among the agents, that has to averaged over 100 consecutive episodes.

## Getting Started
The environment dependencies can be found in the [Udacity Deep Reinforcement Learning Github](https://github.com/udacity/deep-reinforcement-learning#dependencies).
By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

In order to run the Tennis environment a pre built simulation has to be installed according to the specific OS.
The link are reported below:
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Instructions

After installing the local simulator fill in the path to executable in the varible *PATH_TO_TENNIS*  on the first lines of the *main.py*
The agent can then be trained running the *main.py* file in the repo.

The set of pre-train weights of the agent can be found in the *BestWeights* folder. 
A simulation of the agent running the parameters can be checked running *test.py* file. Also in this case, make sure the directory of the executable file is defined in the first line of the file.
