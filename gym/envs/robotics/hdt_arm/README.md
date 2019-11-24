# SIA 7F Arm Mujoco Model

We create a SIA 7F Arm Mujoco Model based on [sia_7f_description](https://github.com/dddsx6324/proj_sia1/tree/master/src/sia_7f_arm_description).

## How to get the mujoco model from a urdf file

You can refer this [link](https://github.com/wangcongrobot/learning-notes/blob/master/mujoco/1-mujoco_modelling.md).

 
## pick and place task

We create a simple pick and place task based on the [Fetch pick and place task](https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py). The original Fetch robotics tasks use a [goal environment](https://arxiv.org/abs/1802.09464), but we change it to a reward shaping environment, which is simpler.

The reward function is from **reward_pick** function:

$r = r_{ctrl} + r_{dist} + r_{grasping}$

in which $r$ is the total reward, $r_{ctrl}$ is a penalty to decrease the action, $r_{dist}$ is inverse propotional to the distance of gripper and object, $r_{grasping}$ is a bonus when get a successful grasping and lifting.

## The file structure

- robotics/assets/stls/sia_7f_arm: mesh file
- robotics/assets/sia_7f_arm: mujoco xml file
- robotics/sia_7f_arm: sia 7f arm pick and place task
- robotics/sia_7f_arm_env.py: gym environment file

## Testing

- install the package:
```bash
$ virtualenv env --python=python3.7
$ souurce env/bin/activate
$ pip install tensorflow==1.14.0 tensorboard==1.14.0 tensorflow-probability==0.7.0 ray[rllib]==0.7.5 requests numpy==1.15.0 mujoco-py==2.0.2.2 psutil 
$ git clone https://github.com/wangcongrobot/gym.git
$ cd gym/
$ pip install -e .
```
- training in the gym/ folder:
```bash
$ rllib train --run PPO --env SIA7FArmPickAndPlace-v1 --checkpoint-freq 20 --config '{"num_workers": 2}'
$ tensorboard --logdir ~/ray_results/default/
```
- evaluation

run in the gym/ folder:
```bash
$ rllib rollout /path/to/ray_results/default/PPO_SIA7FArmPickAndPlace-v1******/checkpoint_xx/checkpoint_xx --run PPO
```

## Results

The training reward is:

![sia_7f_arm_reward](/gym/envs/robotics/results/sia_7f_arm_pick_place_training_reward.png)

The video:

![video](/gym/envs/robotics/results/sia_7f_arm_pick_place.gif)


## How to get the initial position and rotation of the end effector

1. Comment all of the arm joint 
2. Comment the mocap equality
3. Using the test script with no action step, just render
4. print the body quat of end effector palm

## joint

The joint propoty is needed for the tendon:

damping="1.0" armature="0.001"

## underwater

http://www.mujoco.org/book/index.html#Floating

