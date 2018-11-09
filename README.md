# Udacity-RLND-P2-Coutinuous-Control
This is the second project of deep reinforcement learning nanodegree of Udacity

# Single Arm

## Requirements
python 3.6 <br>pytorch 0.4.1<br>

## import package
```
from unityagents import UnityEnvironment
```
### Then Download the environment
Before running the code, change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Reacher.app"
Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"
Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"
For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```
