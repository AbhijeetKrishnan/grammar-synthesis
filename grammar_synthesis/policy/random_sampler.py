from typing import Optional

import numpy as np

from grammar_synthesis.envs import GrammarSynthesisEnv


class RandomSampler:
    "A simple random policy that selects a random action from the masked action space using action_space.sample()"

    def __init__(self, env: GrammarSynthesisEnv) -> None:
        self._action_space = env.action_space

    def get_action(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        return int(self._action_space.sample(mask=mask))
