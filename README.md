# Udacity-RLND-P2-Coutinuous-Control
This is the second project of deep reinforcement learning nanodegree of Udacity

# Reacher Environment

## Requirements
python 3.6 <br>pytorch 0.4.1<br>

## Environment
### import package
```
from unityagents import UnityEnvironment
```

### Then Download the environment
Before running the code, change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Reacher20.app"<br>
Windows (x86): "path/to/Reacher_Windows_x86/Reacher20.exe"<br>
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher20.exe"<br>
Linux (x86): "path/to/Reacher_Linux/Reacher20.x86"<br>
Linux (x86_64): "path/to/Reacher_Linux/Reacher20.x86_64"<br>
Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher20.x86"<br>
Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher20.x86_64"<br>
For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
Single arm version:
```
env = UnityEnvironment(file_name="Reacher.app")
```
Twenty arms version:
```
env = UnityEnvironment(file_name="Reacher20.app")
```
### Information of the environment
In this environment, a double-jointed arm(20 arms in second version) can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To Solve this environment, your agent must get an average score of +30 over 100 consecutive episode

## Code
### To train the agent, use the command below
```
python train_reacher.py
```
### To watch a smart trained agent, please use the command below
```
python infer_reacher.py
```
### Code and files location
The code of the agent is in agent_single_arm.py, the pytorch model is in model_reacher.py<br>
The plot of trainig scores is in ./pictures/reacher.png<br>
The trained model is in ./ckpt/checkpoint1.pth<br>
