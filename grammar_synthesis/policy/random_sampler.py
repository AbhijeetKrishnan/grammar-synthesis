class RandomSampler:
    "A simple random policy that selects a random action from the masked action space using action_space.sample()"

    def __init__(self, env):
        self._action_space = env.action_space

    def get_action(self, obs, mask=None):
        return self._action_space.sample(mask=mask)
