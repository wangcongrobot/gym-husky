import os
from gym import utils
from gym.envs.robotics import sia_7f_arm_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('sia_7f_arm', 'sia_7f_arm.xml')


class SIA7FArmPickAndPlaceEnv(sia_7f_arm_env.SIA7FArmEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.,
            'robot0:slide2': 0.3,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        sia_7f_arm_env.SIA7FArmEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.1, target_range=0.1, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=4)
        utils.EzPickle.__init__(self)
