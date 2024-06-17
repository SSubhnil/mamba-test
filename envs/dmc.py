import gymnasium as gym
import numpy as np
from envs.dmc_meta import suite


class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, detach_image_from_obs=False, small_state_space=False, environment_kwargs={}):
        domain, task = name.split("_", 1)
        self._detach_image_from_obs = detach_image_from_obs
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            self._env = suite.load(domain, task, environment_kwargs=environment_kwargs)
        else:
            assert task is None
            self._env = domain()

        self._env.reset()

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        self._state_keys_to_ignore = ["force_torque"] if small_state_space else []

        default_params = {
            'force_type': 'step',
            'timing': 'random',
            'body_part': 'torso',
            'force_magnitude': 150,  # redundant for now
            'interval': 90,  # redundant for now
            'random_chance': 0.8,  # Chance to apply random force
            'force_range': (90, 170),
            'interval_mean': 90,  # Mean for sampling interval 90, 180
            'interval_std': 10,  # Standard deviation for sampling interval
            'duration_min': 5,  # Minimum duration for swelling force
            'duration_max': 20  # Maximum duration for the swelling force
        }
        self.confounder_params = default_params

        # Initialize attributes based on confounder_params
        self.force_type = self.confounder_params['force_type']
        self.timing = self.confounder_params['timing']
        self.body_part = self.confounder_params['body_part']
        self.force_magnitude = self.confounder_params['force_magnitude']
        self.interval = self.confounder_params['interval']
        self.random_chance = self.confounder_params['random_chance']
        self.force_range = self.confounder_params['force_range']
        self.interval_mean = self.confounder_params['interval_mean']
        self.interval_std = self.confounder_params['interval_std']
        self.duration_min = self.confounder_params['duration_min']
        self.duration_max = self.confounder_params['duration_max']
        self.time_since_last_force = 0
#Logger
    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            if key in self._state_keys_to_ignore:
                continue
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        if not self._detach_image_from_obs:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        self.apply_force()
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items() if key not in self._state_keys_to_ignore}
        if not self._detach_image_from_obs:
            obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()

        done = time_step.last()

        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items() if key not in self._state_keys_to_ignore}
        if not self._detach_image_from_obs:
            obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        scene_image = self._env.physics.render(*self._size, camera_id=self._camera)

        return scene_image

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self._env, item)

    def apply_force(self):
        if self.timing == 'random':
            self.interval = max(30, int(np.random.normal(self.interval_mean,
                                                         self.interval_std)))
            if np.random.uniform() > self.random_chance:
                return

        # Update the timing
        self.time_since_last_force += 1
        if self.time_since_last_force < self.interval:
            return

        # Reset timing for next force application
        self.time_since_last_force = 0

        # Sample the force magnitude fom a normal distribution within the range
        force_magnitude = np.clip(np.random.normal((self.force_range[0] + self.force_range[1]) / 2,
                                                   (self.force_range[1] - self.force_range[0]) / 6),
                                  self.force_range[0], self.force_range[1])

        # Calculate the duration for the force application if 'swelling'
        duration = np.random.randint(self.duration_min, self.duration_max + 1)

        # FLipping the direction for additional challenge
        direction = np.random.choice([-1, 1])

        # Apply swelling or other dynamics based on force type
        # Construct the force vector
        if self.force_type == 'step':
            force = np.array([direction * force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = duration / 2
            # Calculate the standard deviation to control thh width of the bell curve
            sigma = duration / 6  # Adjust as needed for the desired width
            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([direction * magnitude, 0, 0, 0, 0, 0])

        body_id = self._env.physics.model.name2id(self.body_part, 'body')
        # Apply the force
        self._env.physics.data.xfrc_applied[body_id] = force
