from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from grammar_synthesis.envs import GrammarSynthesisEnv


class RandomSampler:
    "A simple random policy that selects a random action from the masked action space using action_space.sample()"

    def __init__(self, env: GrammarSynthesisEnv) -> None:
        self._action_space = env.action_space

    def get_action(
        self,
        obs: NDArray[np.uintc],
        mask: np.ndarray[Tuple[int], np.dtype[np.int8]] | None = None,
    ) -> np.ulonglong:
        return self._action_space.sample(mask=mask)
