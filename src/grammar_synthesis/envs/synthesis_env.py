from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import gymnasium
import numpy as np
import parglare
import parglare.grammar
from gymnasium.spaces import Discrete, MultiDiscrete
from numpy.typing import NDArray

LEGAL = 1
ILLEGAL = 0

ObsType: TypeAlias = NDArray[np.uintc]
ActType: TypeAlias = np.ulonglong


class GrammarSynthesisEnv(gymnasium.Env[ObsType, ActType]):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        grammar: str,
        reward_fn: Callable[[str, Dict[str, Any] | None], float],
        max_len: int = 100,
        render_mode: str | None = "human",
        truncation_reward: float = 0.0,
        mdp_config: Dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes a grammar synthesis environment.

        Args:
            grammar (str):
                The grammar used for synthesis
            reward_fn (Callable[[str, Dict[str, Any]], float]):
                The reward function for the MDP from finished program used as policy.
            max_len (int, optional):
                The maximum allowed sequence length.
                Defaults to `100`
            render_mode (str, optional):
                The rendering mode.
                Defaults to `None`
            truncation_reward (float, optional):
                The reward for exceeding the maximum length.
                Defaults to `0.0`
            mdp_config (Dict[str, Any], optional):
                Secondary MDP config arguments.
                Defaults to `None`
        """

        self.reward_fn = reward_fn
        self.max_len = max_len
        self.truncation_reward = truncation_reward
        self.mdp_config = mdp_config

        self.grammar = parglare.Grammar.from_string(grammar)

        # parglare adds an extra production 0: S' -> [start_symbol]
        self.rules = self.grammar.productions[1:]
        self.num_rules = len(self.rules)
        self.terminals = sorted(
            [
                value
                for key, value in self.grammar.terminals.items()
                if key not in ("STOP", "EMPTY")
            ],
            key=lambda x: x.name,
        )
        self.nonterminals = sorted(
            [
                value
                for key, value in self.grammar.nonterminals.items()
                if key not in ("S'")
            ],
            key=lambda x: x.name,
        )
        self.vocabulary_size = (
            len(self.terminals) + len(self.nonterminals) + 1
        )  # for padding token
        self.vocabulary = {
            token: id
            for (token, id) in zip(
                self.terminals + self.nonterminals, range(1, self.vocabulary_size)
            )
        }

        # list of tokens in the current state
        self.symbols: List[parglare.grammar.GrammarSymbol] = []

        """
        Observations

        A state is a list of tokens consisting of non-terminals and terminals in the leaf nodes of the partial parse tree
        """
        self.observation_space = MultiDiscrete(  # type: ignore
            [self.vocabulary_size] * max_len, dtype=np.uintc
        )  # could use spaces.Sequence or tokenizers pkg

        """
        Actions

        An action is a single production rule applied to a single non-terminal in the current state
        """
        self.action_space = Discrete(  # type: ignore
            self.num_rules * self.max_len
        )  # could use spaces.Sequence but length needs to be equal to sequence across obs and actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self) -> ObsType:
        "Construct observation from environment state"

        return np.pad(
            np.array([self.vocabulary[token] for token in self.symbols], dtype=np.int8),
            (0, max(0, self.max_len - len(self.symbols))),
        )

    def _get_info(self, is_valid: bool = True) -> Dict[str, Any]:
        "Obtain auxiliary info returned by `step` and `reset`"

        return {"action_mask": self.get_action_mask(), "is_valid": is_valid}

    def _repr_state(self) -> str:
        def nt_str(non_terminal: parglare.grammar.NonTerminal) -> str:
            # underline non-terminal when printing to terminal
            return f"\033[4m{non_terminal.name}\033[0m"

        def t_str(terminal: parglare.grammar.Terminal) -> str:
            return str(terminal.name)

        def symbol_str(symbol: parglare.grammar.GrammarSymbol) -> str:
            if isinstance(symbol, parglare.grammar.NonTerminal):
                return nt_str(symbol)
            else:
                return t_str(symbol)

        return " ".join(symbol_str(symbol) for symbol in self.symbols)

    @property
    def program_text(self) -> str:
        return self._repr_state()

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        "Initiate a new episode"

        super().reset(seed=seed, options=options)
        gen_seed = self.action_space.seed(seed)[0]

        # list of tokens starts with start symbol
        self.symbols = [self.grammar.start_symbol]

        obs = self._get_obs()
        info = self._get_info()
        info["seed"] = gen_seed

        return obs, info

    def get_action_mask(self) -> np.ndarray[Tuple[int], np.dtype[np.int8]]:
        "Return valid action mask for current state"
        # TODO: make more efficient by pre-computing some stuff or using np loops

        mask = np.zeros(
            (self.num_rules * self.max_len), dtype=np.int8
        )  # 0 represents illegal action
        for nt_idx, symbol in enumerate(self.symbols):
            if isinstance(symbol, parglare.grammar.NonTerminal):
                for rule in self.grammar.get_productions(symbol.name):
                    rule_idx = rule.prod_id - 1
                    mask[rule_idx * self.max_len + nt_idx] = LEGAL
        return mask

    def encode_action(self, decoded_action: Tuple[int, int]) -> ActType:
        "Encode a (non-terminal index, rule index) as an action"

        nt_idx, rule_idx = decoded_action
        action = rule_idx * self.max_len + nt_idx
        return np.uint64(action)

    def decode_action(self, action: ActType) -> Tuple[int, int]:
        "Decode an action as (non-terminal index, rule index)"

        rule_idx, nt_idx = (
            int(action) // self.max_len,
            int(action) % self.max_len,
        )
        return nt_idx, rule_idx

    def is_valid(self, action: ActType) -> bool:
        "Test if an action is valid"

        nt_idx, rule_idx = self.decode_action(action)

        return (
            nt_idx < len(self.symbols)
            and rule_idx < len(self.rules)
            and self.symbols[nt_idx] == self.rules[rule_idx].symbol
        )

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        nt_idx, rule_idx = self.decode_action(action)

        # if valid action, replace existing non-terminal with rule expansion
        if is_valid := self.is_valid(action):
            self.symbols[nt_idx : nt_idx + 1] = list(self.rules[rule_idx].rhs)
        else:
            print(
                f"Attempted invalid action {action} with NT idx {nt_idx} and rule idx {rule_idx}"
            )

        terminated = all(symbol in self.terminals for symbol in self.symbols)
        truncated = len(self.symbols) > self.max_len
        if truncated:
            self.symbols = self.symbols[: self.max_len]

        if (
            terminated
        ):  # get reward from external MDP by using finished program as policy
            program_text = self._repr_state()
            reward = self.reward_fn(program_text, self.mdp_config)
        elif truncated:  # partial program with len > max_len; cannot be used as policy
            reward = self.truncation_reward
        else:  # partial program; cannot be used as policy
            # TODO: allow specifying a custom function to evaluate a partial program
            reward = 0.0
        obs = self._get_obs()
        info = self._get_info(is_valid=is_valid)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            print(self._repr_state())

    def close(self) -> None:
        pass
