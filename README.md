# Udacity-RLND-P2-Coutinuous-Control
This is the second project of deep reinforcement learning nanodegree of Udacity

# Single Arm Environment

## Requirements
python 3.6 <br>pytorch 0.4.1<br>

## Environment
### import package
```
from unityagents import UnityEnvironment
```

### Then Download the environment
Before running the code, change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Reacher.app"<br>
Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"<br>
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"<br>
Linux (x86): "path/to/Reacher_Linux/Reacher.x86"<br>
Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"<br>
Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"<br>
Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"<br>
For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```
### Information of the environment

## Code
### To train the agent, use the command below
```
python train_single_arm.py
```
### To do inference and see the agent, please use the command below
```
python infer_single_arm.py
```
### Code and files location
The code of the agent is in agent_single_arm.py, the pytorch model is in model_single_arm.py<br>
The plot of trainig scores is in ./pictures/single_arm.png<br>
The trained model is in ./ckpt/checkpoint1.pth<br>
Report of algorithm is in Report of P2_single_arm.pdf<br>
