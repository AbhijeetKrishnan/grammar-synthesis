from typing import Tuple, Callable, Union

import gymnasium
import parglare
import parglare.grammar
import numpy as np


class GrammarSynthesisEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, grammar: str, reward_fn: Callable[[str], float], max_len: int=100, 
                 render_mode=None, truncation_reward: float=0.0, mdp_config=None):
        
        self.reward_fn = reward_fn # reward from MDP from finished program used as policy
        self.max_len = max_len # max allowed sequence length
        self.truncation_reward = truncation_reward # reward for exceeding max length
        self.mdp_config = mdp_config # secondary MDP config arguments

        self.grammar = parglare.Grammar.from_string(grammar)

        self.rules = self.grammar.productions[1:] # parglare adds an extra production 0: S' -> [start_symbol]
        self.num_rules = len(self.rules)
        self.terminals = sorted([value for key, value in self.grammar.terminals.items() if key not in ('STOP', 'EMPTY')], key=lambda x: x.name)
        self.nonterminals = sorted([value for key, value in self.grammar.nonterminals.items() if key not in ("S'")], key=lambda x: x.name)
        self.vocabulary_size = len(self.terminals) + len(self.nonterminals)
        self.vocabulary = {token: id for (token, id) in zip(self.terminals + self.nonterminals, range(self.vocabulary_size))}

        self.symbols = [] # list of tokens in the current state

        """
        Observations
        
        A state is a list of tokens consisting of non-terminals and terminals in the leaf nodes of the partial parse tree
        """
        self.observation_space = gymnasium.spaces.MultiDiscrete([self.vocabulary_size] * max_len) # could use spaces.Sequence or tokenizers pkg

        """
        Actions

        An action is a single production rule applied to a single non-terminal in the current state
        """
        self.action_space = gymnasium.spaces.Discrete(self.num_rules * self.max_len) # could use spaces.Sequence but length needs to be equal to sequence across obs and actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self) -> np.ndarray:
        "Construct observation from environment state"

        return np.pad(np.array([self.vocabulary[token] for token in self.symbols]), (0, max(0, self.max_len - len(self.symbols))))

    def _get_info(self, is_valid=True):
        "Obtain auxiliary info returned by `step` and `reset`"

        return {"action_mask": self.get_action_mask(), 'is_valid': is_valid}
    
    def _repr_state(self):
        return ' '.join(str(symbol.name) for symbol in self.symbols)

    def reset(self, seed=None, options=None):
        "Initiate a new episode"

        super().reset(seed=seed)

        self.symbols = [self.grammar.start_symbol] # list of tokens starts with start symbol

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def get_action_mask(self) -> np.ndarray:
        "Return valid action mask for current state"
        # TODO: make more efficient by pre-computing some stuff or using np loops

        mask = np.array([0] * (self.num_rules * self.max_len), dtype=np.int8)
        for nt_idx, symbol in enumerate(self.symbols):
            if type(symbol) == parglare.grammar.NonTerminal:
                for rule in self.grammar.get_productions(symbol.name):
                    rule_idx = rule.prod_id - 1
                    mask[rule_idx * self.max_len + nt_idx] = 1 # if 0, action is masked (illegal), else not masked (legal)
        return mask
    
    def encode_action(self, decoded_action: Tuple[int, int]) -> int:
        "Encode a (non-terminal index, rule index) as an action"
        
        nt_idx, rule_idx = decoded_action
        action = rule_idx * self.max_len + nt_idx
        return action

    def decode_action(self, action: int) -> Tuple[int, int]:
        "Decode an action as (non-terminal index, rule index)"
        rule_idx, nt_idx = action // self.max_len, action % self.max_len
        return nt_idx, rule_idx
    
    def is_valid(self, action: Union[int, Tuple[int, int]]) -> bool:
        "Test if an action is valid"

        if type(action) == int:
            nt_idx, rule_idx = self.decode_action(action)
        else:
            nt_idx, rule_idx = action
        return (
            nt_idx < len(self.symbols) and 
            rule_idx < len(self.rules) and 
            self.symbols[nt_idx] == self.rules[rule_idx].symbol
        )

    def step(self, action: Union[int, Tuple[int, int]]):
        
        if type(action) == int:
            nt_idx, rule_idx = self.decode_action(action)
        else:
            nt_idx, rule_idx = action
        if self.is_valid(action): # if valid action, replace existing non-terminal with rule expansion
            self.symbols[nt_idx:nt_idx+1] = list(self.rules[rule_idx].rhs)
        else:
            print(f'Attempted invalid action {action} with NT idx {nt_idx} and rule idx {rule_idx}')
            is_valid = False

        terminated = all(symbol in self.terminals for symbol in self.symbols)
        truncated = len(self.symbols) > self.max_len
        if truncated:
            self.symbols = self.symbols[:self.max_len]

        if terminated: # get reward from external MDP by using finished program as policy
            program_text = self._repr_state()
            reward = self.reward_fn(program_text, self.mdp_config)
        elif truncated: # partial program with len >= max_len; cannot be used as policy
            reward = self.truncation_reward
        else: # partial program; cannot be used as policy
            reward = 0.0 # sum(-1.0 for symbol in self.symbols if type(symbol) == lark.grammar.NonTerminal)
        obs = self._get_obs()
        info = self._get_info(is_valid=is_valid)

        return obs, reward, terminated, truncated, info

    def render(self):  
        print(self._repr_state())

    def close(self):
        pass
