import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SIA7FArmEnv(robot_env.RobotEnv):
    """Superclass for all SIA 7F Arm environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, n_actions,
    ):
        """Initializes a new Dual_UR5_Husky environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            n_actions : the number of actuator
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_actions = n_actions # the total action number

        self.arm_dof = 3 # (x,y,z)
        self.gripper_dof = 1 # open/close
        
        self.gripper_actual_dof = 6 # 6 joint in gripper xml file
        self.gripper_close = False

        super(SIA7FArmEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, action, goal):
        return self.reward_pick(action, goal)

    def reward_pick(self, action, goal):
        """
        Simple reward function: reach and pick
        """

        object_pos = self.sim.data.get_site_xpos('object0')
        print("self.sim.data.get_site_xpos('object0'): ", object_pos)
        grip_pos = self.sim.data.get_site_xpos('r_grip_site')
        print("self.sim.data.get_site_xpos('r_grip_pos'): ", grip_pos)

        grip_obj_pos = object_pos - grip_pos
        obj_target_pos = goal - object_pos

        reward_ctrl = 0
        reward_dist_object = 0
        reward_grasping = 0
        reward_dist_target = 0
        reward_target = 0
        reward = 0

        reward_ctrl = -np.square(action).sum()

        reward_dist_object = -np.linalg.norm(grip_obj_pos)
        print("distance between gripper and object: ", reward_dist_object)
        reward_dist_target = -np.linalg.norm(obj_target_pos)

        self.gripper_close = False # this flag for gripper open/close trigger
        if np.linalg.norm(grip_obj_pos) < 0.05:
            # reward_grasping += 1.0 # it may cause a local minimum
            self.gripper_close = True
            if object_pos[2] > 0.75: # the default height of object on the table
                # grasping success
                reward_grasping += 10.0
                if object_pos[2] > 0.8: # lifting 
                    reward_grasping += 100.0

        reward = 0.01 * reward_ctrl + reward_dist_object + reward_grasping

        print("object_pose: ", object_pos)
        print("reward_dist_object: ", reward_dist_object)
        print("reward_ctrl: ", 0.05 * reward_ctrl)
        print("reward_grasping: ", reward_grasping)
        # print("reward_dist_target: ", reward_dist_target)
        # print("reward_target: ", reward_target)
        print("total reward: ", reward)
        done = False
        if object_pos[2] < 0.1: # avoid the object dropped on the ground
            # done = True
            reward -= 10
        return reward, done


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        print("_set_action:", action)
        pos_ctrl, gripper_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.03  # limit maximum change in position
        rot_ctrl = [0.5, 0.5, -0.5, -0.5] # # fixed rotation of the end effector, expressed as a quaternion

        # you can comment this part to use a random gripper control
        if self.gripper_close:
            gripper_ctrl = 1.0
        else:
            gripper_ctrl = -1.0

        gripper_ctrl = self.gripper_format_action(gripper_ctrl)
        assert gripper_ctrl.shape == (self.gripper_actual_dof,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action) # gripper control
        utils.mocap_set_action(self.sim, action) # arm control in cartesion (x, y, z)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('r_grip_site')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('r_grip_site') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        # gripper_state = robot_qpos[-2:]
        gripper_state = robot_qpos[-13:-1]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('r_gripper_palm_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            # the inital position of (x,y) plus a random value
            object_xpos = np.array([1.0, -0.0]) + self.np_random.uniform(-0.02, 0.07, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            print("object_xpos0: ", object_xpos)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.9 # the initial height of object

            print("set_joint_qpos object_qpos: ", object_qpos)
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # print("get_body_xquat: ", self.sim.data.get_body_xquat('r_gripper_palm_link'))

        # give a ramdom position for the gripper
        gripper_target = np.array([0.80326763, 0.01372008, 0.7910795])
        gripper_rotation = np.array([0.5, 0.5, -0.5, -0.5]) # fixed gripper rotation
        # random value for x,y,z
        gripper_target[0] += self.np_random.uniform(-0.0, 0.1) # x
        gripper_target[1] += self.np_random.uniform(-0.1, 0.1) # y
        gripper_target[2] += self.np_random.uniform(-0., 0.1) # z
        self.sim.data.set_mocap_pos('gripper_r:mocap', gripper_target)
        self.sim.data.set_mocap_quat('gripper_r:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0.2, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector to initial position.
        print("gripper quat: ", self.sim.data.get_site_xmat('r_grip_site'))
        print("get_mocap_quat: ", self.sim.data.get_mocap_quat('gripper_r:mocap'))

        gripper_target = np.array([0.80326763, 0.01372008, 0.7910795]) 
        gripper_rotation = np.array([0.5, 0.5, -0.5, -0.5]) #(0, 0, -90)

        self.sim.data.set_mocap_pos('gripper_r:mocap', gripper_target)
        self.sim.data.set_mocap_quat('gripper_r:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('r_grip_site').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(SIA7FArmEnv, self).render(mode, width, height)

    ### Add some useful function

    def gripper_format_action(self, action):
        """ Given (-1,1) abstract control as np-array return the (-1,1) control signals
        for underlying actuators as 1-d np array
        Args:
            action: 1 => open, -1 => closed
        """
        movement = np.array([-1, -1, -1, 1, 1, 1])
        return -1 * movement * action



    