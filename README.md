[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

An Double Deep Q Learning Agent (DDQN Agent), with prrioritized replay buffer, has been trained to navigate (and collect bananas!) in a large, square world.

## Goal of the project
![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
### State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  

### Action Space

Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Solution

A **Double DQN** agent with **Prioritized Replay Buffer** is used for solving the problem.

### Double DQN Agent Algorithm

When a single neural network is used as the Q function approximator, reinforcement learning is known to be unstable or even diverge from the solution. One of the issues is that a small update to the network (that may result from one back propogration cycle for an experience) may significantly change the policy, which will impact all future Q(S', A). The second is the strong correlation that exist between sequence of observations.

To overcome the latter issue, DQN uses a replay buffer, in which the experience that the agent gather's from acting in the environment is stored as a tupe of (state, action, reward, next_state and dones). Each training step, the agent randomly samples a batch of k size from the replay buffer and uses that for training. Initially, the agent waits for the replay buffer to have atleast N steps in it before it does any training.

For the update issue, the other thing DQN Agent does is to use two similar shaped neural networks instead of one. One is called the Local (or online) network, represented by θ and the other is called Target network represented by θ′. Each time step, the agent uses the local network to choose an action in a ϵ greedy fashion. From training perspective, the local network is trained and the target network is used for comparing the local network's output with the desired outcome. Every C steps the online network is copied over to the target network. The loss function used for training is:

*Add image here for loss function*

In essence, for finding out the value for the next state, V(S′):

1. S′ is passed through the target Q network
2. From the output of the network, the action that has the maximum value is picked and V(S′) = max action's value

From there the loss function is computed and back propogated into the local network.



A Double DQN Agent, uses the online network to find the best action in `next_state` but uses the value of that particular action from the target network.

The agent used double DQN algorithm. The differen
### Network Architecture


### Prioritized Replay Buffer

[Paper Referencec]()

### Hyper Parameters


## Future Work

## Instructions on how to setup 
### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
