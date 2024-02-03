from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from grammar_synthesis.envs import GrammarSynthesisEnv


class UniformRandomSampler:
    "A simple random policy that selects a random action from the masked action space using action_space.sample()"

    def __init__(self, env: GrammarSynthesisEnv) -> None:
        self._action_space = env.action_space

    def get_action(
        self,
        obs: NDArray[np.uintc],
        mask: np.ndarray[Tuple[int], np.dtype[np.int8]] | None = None,
    ) -> np.ulonglong:
        return self._action_space.sample(mask=mask)


class WeightedRandomSampler:
    "A random policy that selects a random action given a list of probabilities for sampling each token"

    def __init__(self, env: GrammarSynthesisEnv, weights: List[float] | None) -> None:
        self._action_space = env.action_space
        self._weights = weights

    def get_action(
        self,
        obs: NDArray[np.uintc],
        mask: np.ndarray[Tuple[int], np.dtype[np.int8]] | None = None,
    ) -> np.ulonglong:
        weights = np.array(self._weights)
        weights[
            mask == 0
        ] = 0  # mask = 1 -> valid, so set probs of invalid actions to 0
        weights /= np.sum(weights)
        return self._action_space.np_random.choice(self._action_space.n, p=weights)  # type: ignore
