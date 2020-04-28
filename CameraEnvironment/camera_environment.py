import numpy as np


class CameraEnvironment(object):
    def __init__(self,failure_chance=0.05,init_seed=None):
        assert 0<=failure_chance<1

        self.state_size = 8
        self.action_size = 2
        self.action_bounds = [(-1,1),(-1,1)]

        self.field_w = 2048 # px
        self.field_h = 1080  # px
        self.frame_size = np.array([16,9])*5 # [px,px]
        self.camera_max_speed = 10  # px/ms
        self.command_period = 50 # in ms
        self.num_points = 4
        self.border_size = 0.1
        self.object_speed_smooth_ratio = 0.99
        self.failure_chance = failure_chance
        self.seed(init_seed)

        self.max_timesteps = 200
        self.done = False

    def seed(self,seed):
        self._random_generator = np.random.RandomState(seed)
        T1 = self._random_generator.uniform(0.15, 0.25)
        T2 = self._random_generator.uniform(0.025, 0.075)
        T3 = self._random_generator.uniform(0.1, 0.75)
        C = self._random_generator.uniform(0.01, 0.25)
        time = np.arange(self.command_period)
        self._transfer = 0.75*np.exp(-T1 * time) + 0.25*np.exp(-T2 * time)*(np.cos(T3*time) + C*np.sin(T3*time))
        self._transfer = self._transfer[:,None]

    def reset(self):
        w = self._random_generator.randint(0,self.field_w)
        h = self._random_generator.randint(0,self.field_h)
        self.camera_position = np.array([w,h],dtype=np.float)
        self.camera_speed = np.zeros((2,),dtype=np.float)

        self.done = False
        self.timestep = 0
        self._time = 0

        num_points = self.num_points
        attempts = 0
        while True:
            points_x = self._random_generator.randint(int(self.border_size * self.field_w), int((1 - self.border_size) * self.field_w), size=num_points)
            points_y = self._random_generator.randint(int(self.border_size * self.field_h), int((1 - self.border_size) * self.field_h), size=num_points)

            t = self.command_period * self.max_timesteps
            time_mat = np.linspace(0, t, num=num_points, dtype=np.float)
            time_mat = np.stack([time_mat ** i for i in range(num_points)], axis=1)
            time_mat_pinv = np.linalg.pinv(time_mat)
            coeff_x = time_mat_pinv.dot(points_x)
            coeff_y = time_mat_pinv.dot(points_y)

            t = np.arange(t+1)
            traj_x = sum(coeff_x[i] * t ** i for i in range(num_points))
            traj_y = sum(coeff_y[i] * t ** i for i in range(num_points))

            out_x_border = np.any(np.logical_or(traj_x <= 0, traj_x >= self.field_w))
            out_y_border = np.any(np.logical_or(traj_y <= 0, traj_y >= self.field_h))
            if not (out_x_border or out_y_border):
                break
            if attempts >= 10:
                num_points -= 1
            attempts += 1
        self.object_trajectory = np.stack([traj_x,traj_y],axis=1)
        self.object_trajectory += self._random_generator.normal(0.0,1,size=self.object_trajectory.shape) # noisy position mesurments
        self.object_speed = np.array([coeff_x[1],coeff_y[1]],dtype=np.float) # fair speed is only for the first moment
        return self._form_state()

    def step(self, action):
        assert action.size == 2
        assert not self.done , "Simulation is finished"

        if self._random_generator.random() < self.failure_chance:
            act = self._random_generator.randint(-1,1,size=(2,)) # Ooops, camera controller sudden insanity
        else:
            act = np.clip(action,-1,1)

        delim = np.array([self.field_w, self.field_h], dtype=np.float) / 2
        camera_positions = np.zeros((self.command_period, 2))
        reward = 0
        camera_speed_transfer = act[None,:] * (1 - self._transfer) + self.camera_speed[None,:] * self._transfer
        camera_speed_transfer = np.clip(camera_speed_transfer, -1, 1) * self.camera_max_speed
        for i in range(self.command_period):
            self._time += 1
            obj_position = self.object_trajectory[self._time]
            self.object_speed = self.object_speed_smooth_ratio * self.object_speed + \
                                (1-self.object_speed_smooth_ratio)*(obj_position - self.object_trajectory[self._time-1]) # no fair speed, only exponential decay
            self.camera_position += camera_speed_transfer[i]
            self.camera_position[0] = np.clip(self.camera_position[0], 0, self.field_w)
            self.camera_position[1] = np.clip(self.camera_position[1], 0, self.field_h)

            camera_positions[i,:] = self.camera_position

            cam_obj_norm_dist = np.abs(self.camera_position/ delim - obj_position/ delim )
            reward -= np.sum(cam_obj_norm_dist)

        self.camera_speed[:] = camera_speed_transfer[-1] / self.camera_max_speed
        if self.camera_position[0] == 0.0 or self.camera_position[0] == self.field_w:
            self.camera_speed[0] = 0.0
        if self.camera_position[1] == 0.0 or self.camera_position[1] == self.field_h:
            self.camera_speed[1] = 0.0

        self.timestep += 1
        self.done = self.timestep == self.max_timesteps

        delim = delim[None, :]
        info = {
            "cam_pos" : camera_positions/ delim - 1,
            "obj_pos" : self.object_trajectory[self._time-self.command_period+1:self._time+1]/ delim - 1
        }
        return self._form_state(), reward, self.done, info

    def _form_state(self):
        delim = np.array([self.field_w,self.field_h],dtype=np.float) / 2
        normed_cam = self.camera_position / delim - 1
        normed_obj = self.object_trajectory[self._time] / delim - 1
        return np.concatenate([normed_cam, self.camera_speed, normed_obj-normed_cam, self.object_speed - self.camera_speed * self.camera_max_speed], axis=0)

    def _get_position_mesurment_noise(self):
        return self._random_generator.normal(0.0,1,size=(2,))

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
