[//]: # (Image References)

[image1]: /assets/demo.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


# Project 1: Navigation

## Introduction

A Double Deep Q Learning Agent (DDQN Agent), with prrioritized replay buffer, has been trained to navigate (and collect bananas!) in a large, square world.

## Goal of the project

![Trained Agents][image1]

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

**Saved Network Weights**: qnetwork.pt

**Episodes Required**: Around 700 to 900 (multiple runs of the same code has results in different number of episodes)
### DQN Algorithm

When a single neural network is used as the Q function approximator, reinforcement learning is known to be unstable or even diverge from the solution. One of the issues is that a small update to the network (that may result from one back propogration cycle for an experience) may significantly change the policy, which will impact all future Q(S', A). The second is the strong correlation that exist between sequence of observations.

To overcome the latter issue, DQN uses a replay buffer, in which the experience that the agent gather's from acting in the environment is stored as a tupe of (state, action, reward, next_state and dones). Each training step, the agent randomly samples a batch of k size from the replay buffer and uses that for training. Initially, the agent waits for the replay buffer to have atleast N steps in it before it does any training.

For the update issue, the other thing DQN Agent does is to use two similar shaped neural networks instead of one. One is called the Local (or online) network, represented by θ and the other is called Target network represented by θ′. Each time step, the agent uses the local network to choose an action in a ϵ greedy fashion. From training perspective, the local network is trained and the target network is used for comparing the local network's output with the desired outcome. Every C steps the online network is copied over to the target network. The loss function used for training is:

In essence, for finding out the value for the next state, V(S′):

1. S′ is passed through the target Q network
2. From the output of the network, the action that has the maximum value is picked and V(S′) is set to max action's value
3. q = r + γ Q<sub>target</sub>(S′, max <sub>target</sub> a′)
4. td_error is defined as q - Q<sub>local</sub>(S, a)
5. loss function is defined as the mean of square of td_error

From there the loss function is computed and derivative of it is back propogated into the local network.

![LossFunction](/assets/dqn-loss-functions.PNG)

### Double DQN (DDQN) Algorithm

Since to choose an action in a given state, the agent uses the local network therefore the DDQN algorithm, uses the online network to find which action would be best to take in **S′ but then it uses the target network to find the value of the max action in S′**.

Changes for DDQN:

1. S′ is passed through the local network, index of the maximum action is found from the output
2. S' is then passed through the target network and the value of the index found in step 1, is used to figure out Q(S′, a′)
3. q = r + γ Q<sub>target</sub>(S′, max_arg Q<sub>local</sub>(S′))

![DLossFunction](/assets/ddqn-y.PNG)


The original paper is available at: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
### Prioritized Replay Buffer

The idea behind prioritized replay buffer is that some states have a much higher td_error and offer a greater opportunity for the agent to learn from them. However, using a normal stochastic replay buffer, there is no gaurantee that such high error states would be chosen often till the error in that state is reduced.

Prioritized Replay Buffer provides a higher chance to a state of being chosen in a given minibatch by recording the td_error with each state and then using that as the priority at the time of choice. Since the learning and acting parts of the agent are separate, at the time the agent acts in the environment, it attaches the highest priority to the experience (state, action, reward, next_state, done) that resulted from the action and records it in the Prioritized Replay Buffer. This way it can gaurantee that each action will be chosen at least once in the learning stages. Since the states that have higher td_error will be chosen more often the resulting neural network will overfit. In order to reduce the chances of overfitting, importance sampling rate (β) is used to reduce the weight updates. 

To record priorities and to assist in choosing experiences based on their priority a **Sum Tree** data structure has been implemented. A Sum Tree is a complete binary tree with the leaf nodes being the experiences that have been recorded  and the non-leaf nodes are the sum of priority (td_error) of each of the two children nodes. This way the root of the tree has the sum of all of the experiences that have been recorded in the Prioritized Replay Buffer. The absolute value of the td_error is used as priority.

To choose an experience at random from the Sum Tree, a uniform random number is generated between the range 0 to Sum of Priorities (value of the root node of the tree). Then the tree is traversed using the following algorithm:

1. Keep traversing the tree until a leaf node is reached
2. Set p to the root node's value
3. At each node compare p with the value of left child of the node.
4. If p is less than the left node's value, go to the left child.
5. If p is greater than the left node's value, subtract left node's value from p and go to the right node
6. When a leaf node is reached, return that as the chosen experience

[The test notebook](test/test_sumtree.ipynb) proves that if sufficent large number of times the above steps are followed to choose numbers from a Sum Tree then the overall distribution of how many times a particular number was chosen would reflect its priority amongst all numbers added to the tree.

![Weights](/assets/weights.PNG)


### Double DQN with Prioritized Replay Buffer

Some additional steps are required to be carried out for implementing Double DQN with Prioritized Replay Buffer. These steps include the following changes:

. Loss function is to be multipled by importance sampling weights, which are defined using:

1. k experiences are chosen from the Sum Tree

2. A hyper parameter α is defined. P(J) is defined as each chosen experience's priority <sup>α</sup> / (sum of all chosen experiences's priority) <sup>α</sup>

3. A hyper parameter β is defined and importance sample weights are computed as (1 / N * 1 / P(J))<sup>β</sup>

4. Loss function is defined as the mean of (td_error * importance sampling weights)<sup>2</sup>.


![Algo](/assets/algo.PNG)


The original paper is available at: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

### Network Architecture

The network consists of 2 Linear layers with Relu as the activation function:

Sequential(    
  (0): Linear(in_features=37, out_features=64, bias=True)    
  (1): ReLU()    
  (2): Linear(in_features=64, out_features=32, bias=True)    
  (3): ReLU()    
  (4): Linear(in_features=32, out_features=4, bias=True)    
) 

![Viz](/assets/viz.PNG)

### Hyper Parameters

|Parameter|Value|Description|
|-|-|-|
|Min Replay Buffer|10,000|Experiences are only recorded for at least these many steps|
|Mini Batch Size|64|Each time learning is carried out these many experiences are taken from the replay buffer|
|Total Buffer Size|100,000|These many experiences are always kept in the buffer|
|Optimizer Used|Adam||
|LR (learning Rate)|0.0005|Adam optimizer learning rate|
|Gamma|0.99|Discount rate for computing future rewards values|
|Tau|0.001|Weightage of copying online parameters to target. θ_target = τ*θ_local + (1 - τ)*θ_target |
|ϵ|Annealed from 0.98 down to 0.01|Agent takes a random action ϵ greedy times. This helps in keeping a balance between exploitation and exploration. The annealing starts after 20,000 steps and ends at 100,000|
|L steps|4|Every 4 steps the agent learns|
|C steps|12|Every 12th step rates are copied from local to target|
|α|0.6|Probablity factor used as exponent for priority given to each experience|
|β|0.4 down to 1.0|Used in computing the importance-sampling rate. Starts annealing after 200,000 steps and ends at 50,000,000|


## Training Plot

The plot for the overll training is:

![trainingplot](/assets/plot-overall.png)

The plot for the last 100 episodes is:

![trainingplot](/assets/plot-100.png)


## Future Work

1. The replay buffer uses round robin strategy for recording new experiences. It can be enhanced to replace the minimum priority experience each time a new one is to be recorded

2. Duel DQN can be implemented

3. How does training change with different values of α and β is unclear and needs to be looked into.

4. Try to solve the environment by using raw screen pixels.

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

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]
