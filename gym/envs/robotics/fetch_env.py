import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

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

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=6,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward_old(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # def step(self, a):
    #     ball_xy = self.get_body_com("ball")[:2]
    #     goal_xy = self.get_body_com("goal")[:2]

    #     if not self._ball_hit_ground and self.get_body_com("ball")[2] < -0.25:
    #         self._ball_hit_ground = True
    #         self._ball_hit_location = self.get_body_com("ball")

    #     if self._ball_hit_ground:
    #         ball_hit_xy = self._ball_hit_location[:2]
    #         reward_dist = -np.linalg.norm(ball_hit_xy - goal_xy)
    #     else:
    #         reward_dist = -np.linalg.norm(ball_xy - goal_xy)
    #     reward_ctrl = - np.square(a).sum()

    #     reward = reward_dist + 0.002 * reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist,
    #             reward_ctrl=reward_ctrl)

    def compute_reward1(self, action, goal):
        # control_mult = 0.1
        # r_control = (1 - np.tanh(np.square(action).sum())) * control_mult
        r_control = -0.01 * np.square(action).sum()
        print("r_control: ", r_control)

        staged_reward = self.staged_reward(action, goal)
        # print("staged_reward: ", staged_reward)
        reward = r_control + staged_reward
        done = False
        object_pos = self.sim.data.get_site_xpos('object0')
        if object_pos[2] < 0.2:
            done = True
        print("total reward: ", reward)
        return reward, done

    def compute_reward(self, action, goal):
        return self.reward_pick(action, goal)
        # return self.reward_place(action, goal)
        # return self.staged_reward_simple(action, goal)
        # return self.staged_reward_standford(action, goal)

    def staged_reward_simple(self, action, goal):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """
        object_pos = self.sim.data.get_site_xpos('object0')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_obj_pos = object_pos - grip_pos
        obj_target_pos = goal - object_pos
        
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7
        target_mult = 0.9
        reward = 0.

        ### reaching reward governed by distance to closest object ###
        r_reach = 0
        # get reaching reward via minimum distance to a target object
        dist = np.linalg.norm(grip_obj_pos)
        # r_reach = (1 - np.tanh(1.0 * dist)) * reach_mult
        r_reach = - dist * reach_mult

        ### grasping reward for touching any objects of interest ###
        r_grasp = 0.
        # touch_left_finger = False
        # touch_right_finger = False
        # touch_object = False
        # if np.linalg.norm(grip_obj_pos) < 0.05:
            # touch_object = True
        
        if object_pos[2] > (0.4 + 0.025 + 0.05):
            r_grasp = grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        # if r_grasp > 0.:
            # object_pos[2] > (0.4 + 0.025 + 0.05)
            # r_lift = lift_mult
            # z_dist = 0.5 - object_pos[2]
            # r_lift = grasp_mult + (1 - np.tanh(z_dist)) * (lift_mult - grasp_mult)

        ### hover reward for getting object to target ###
        r_hover = 0.
        # if r_lift > 0.:
        #     dist_hover = np.linalg.norm(obj_target_pos)
        #     r_hover = grasp_mult + (1 - np.tanh(dist_hover)) * (hover_mult - grasp_mult)
        
        if r_grasp > 0.: 
            dist_hover = np.linalg.norm(obj_target_pos)
            r_hover = grasp_mult + (1 - np.tanh(dist_hover)) * (hover_mult - grasp_mult)

        ### target ###
        r_target = 0.
        # if r_hover > 0.:
        dist_target = np.linalg.norm(obj_target_pos)
        if dist_target < 0.05:
            r_target = target_mult

        if r_grasp > 0.:
            reward = r_hover + r_target
        else:
            reward = r_reach + r_grasp
        # staged_reward = r_reach + r_grasp + r_hover + r_target
        print("reward_reach: ", r_reach)
        print("reward_grasp: ", r_grasp)
        print("reward_hover: ", r_hover)
        print("reward_target: ", r_target)

        done = False
        if object_pos[2] < 0.2:
            done = True
        return reward, done
        # return r_reach, r_grasp, r_lift, r_hover, r_target

    def staged_reward_standford(self, action, goal):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """
        object_pos = self.sim.data.get_site_xpos('object0')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_obj_pos = object_pos - grip_pos
        obj_target_pos = goal - object_pos
        
        control_mult = 0.1
        reach_mult = 0.1
        grasp_mult = 0.5
        lift_mult = 0.35
        hover_mult = 0.7
        target_mult = 0.9
        reward = 0.

        ### control action ###
        action_sum = np.square(action).sum()
        r_ctrl = (1 - np.tanh(0.5 * action_sum)) * control_mult

        ### reaching reward governed by distance to closest object ###
        r_reach = 0
        # get reaching reward via minimum distance to a target object
        dist = np.linalg.norm(grip_obj_pos)
        # r_reach = (1 - np.tanh(1.0 * dist)) * reach_mult
        r_reach = - dist * reach_mult
        # r_reach = - dist * reach_mult

        ### grasping reward for touching any objects of interest ###
        # r_grasp = 0.
        # touch_left_finger = False
        # touch_right_finger = False
        # touch_object = False
        # if np.linalg.norm(grip_obj_pos) < 0.05:
            # r_grasp = grasp_mult
            # touch_object = True
        # if object_pos[2] > (0.4 + 0.025 + 0.05):
            # r_grasp = grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        # if r_grasp > 0. and np.linalg.norm(grip_obj_pos) < 0.04:
            # object_pos[2] > (0.4 + 0.025 + 0.05)
            # r_lift = lift_mult
        print("grip_obj_pos: ", np.linalg.norm(grip_obj_pos))
        if np.linalg.norm(grip_obj_pos) < 0.05:
            z_dist = 0.5 - object_pos[2]    
            print("object_pos_z: ", object_pos[2])
            print("z_dist: ", z_dist)
            # r_lift = reach_mult + (1 - np.tanh(1.0 * np.linalg.norm(z_dist))) * (lift_mult - reach_mult)

        # r_lift = 0.
        # if object_pos[2] > (0.4 + 0.025 + 0.05):
            # r_lift = lift_mult

        r_grasp = 0.
        print("object_pos[2]: ", object_pos[2])
        if object_pos[2] > (0.4 + 0.025 + 0.02):
            r_grasp = grasp_mult

        ### hover reward for getting object to target ###
        r_hover = 0.
        if r_grasp > 0.:
            dist_hover = np.linalg.norm(obj_target_pos)
            print("dist_hover: ", dist_hover)
            r_hover = grasp_mult + (1 - np.tanh(1.0 * dist_hover)) * (hover_mult - grasp_mult)

        ### target ###
        r_target = 0.
        # if r_hover > 0.:
        dist_target = np.linalg.norm(obj_target_pos)
        if dist_target < 0.1:
            r_target = target_mult
            if dist_target < 0.05:
                r_target += target_mult
                if dist_target < 0.01:
                    r_target += target_mult

        print("reward_control: ", r_ctrl)
        print("reward_reach: ", r_reach)
        print("reward_lift: ", r_lift)
        print("reward_grasp: ", r_grasp)
        print("reward_hover: ", r_hover)
        print("reward_target: ", r_target)
        staged_reward = [r_reach, r_grasp, r_lift, r_hover, r_target]
        # staged_reward = [r_reach, r_grasp, r_lift]
        print("staged_reward: ", staged_reward)

        reward = r_ctrl + max(staged_reward)
        print("total reward: ", reward)

        done = False
        if object_pos[2] < 0.2:
            done = True
        
        return reward, done
        # return r_reach, r_grasp, r_lift, r_hover, r_target

    def reward_pick(self, action, goal):
        """
        Simple reward function: reach and pick
        """
        object_pos = self.sim.data.get_site_xpos('object0')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
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
        reward_dist_target = -np.linalg.norm(obj_target_pos)

        # stage 1: approaching and grasping
        if object_pos[2] > (0.4 + 0.025 + 0.05): # table hight + object hight + lift distance
            # grasping success
            reward_grasping = 10

        reward = 0.05 * reward_ctrl + reward_dist_object + reward_grasping

        # stage 2: approaching and target
        # if reward_grasping > 0:
            # if np.linalg.norm(obj_target_pos) < 0.05:
                # reward_target = 20
            # reward = 0.05 * reward_ctrl + reward_dist_target + reward_target
            
        # reward = 0.05 * reward_ctrl + reward_dist_object + reward_grasping + 10 * reward_dist_target + reward_target
        print("object_pose: ", object_pos)
        print("reward_dist_object: ", reward_dist_object)
        print("reward_ctrl: ", 0.05 * reward_ctrl)
        print("reward_grasping: ", reward_grasping)
        # print("reward_dist_target: ", reward_dist_target)
        # print("reward_target: ", reward_target)
        print("total reward: ", reward)
        done = False
        if object_pos[2] < 0.2:
            done = True
        return reward, done

    def reward_place(self, action, goal):
        """
        Simple reward function: reach and pick
        """
        object_pos = self.sim.data.get_site_xpos('object0')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
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
        reward_dist_target = -np.linalg.norm(obj_target_pos)

        # stage 1: approaching and grasping
        if object_pos[2] > (0.4 + 0.025 + 0.05): # table hight + object hight + lift distance
            # grasping success
            reward_grasping = 10
        # reward = 0.05 * reward_ctrl + reward_dist_object + reward_grasping

        # stage 2: approaching and target
        if reward_grasping > 0:
            if np.linalg.norm(obj_target_pos) < 0.1:
                reward_target = 10
                if np.linalg.norm(obj_target_pos) < 0.05:
                    reward_target += 10
                    if np.linalg.norm(obj_target_pos) < 0.01:
                        reward_target += 10

        reward = 0.05 * reward_ctrl + reward_dist_target + reward_target

        # reward = 0.05 * reward_ctrl + reward_dist_object + reward_grasping + 10 * reward_dist_target + reward_target
        print("object_pose: ", object_pos)
        print("reward_dist_object: ", reward_dist_object)
        print("reward_ctrl: ", 0.05 * reward_ctrl)
        print("reward_grasping: ", reward_grasping)
        print("reward_dist_target: ", reward_dist_target)
        # print("reward_target: ", reward_target)
        print("total reward: ", reward)
        done = False
        if object_pos[2] < 0.2:
            done = True
        return reward, done

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (6,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # print("set_action:", action)
        pos_ctrl, base_ctrl, gripper_ctrl = action[:3], action[3:-1], action[-1]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl, base_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
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
        gripper_state = robot_qpos[-2:]
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
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
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
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

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

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
