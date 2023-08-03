from typing import Tuple, Callable

import gymnasium
import lark
import lark.grammar
import numpy as np


class GrammarSynthesisEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, grammar: str, start_symbol: str, reward_fn: Callable[[str], float], max_len: int=100, 
                 render_mode=None, truncation_reward=0.0, parser: str='earley', mdp_config=None):
        self.parser = lark.Lark(grammar, parser=parser, start=start_symbol)
        self.start_symbol = self.parser.rules[0].origin
        self._num_rules = len(self.parser.rules)
        self.max_len = max_len # max allowed sequence length
        self.terminals = [lark.grammar.Terminal(terminal_def.name) for terminal_def in self.parser.terminals]
        self.non_terminals = list({rule.origin for rule in self.parser.rules})
        self.vocabulary = {token: id for (token, id) in zip(self.terminals + self.non_terminals, range(len(self.terminals) + len(self.non_terminals)))}
        self.vocabulary_size = len(self.vocabulary)
        self.symbols = []
        self.reward_fn = reward_fn # reward from MDP from finished program used as policy
        self.mdp_config = mdp_config # secondary MDP config arguments
        self.truncation_reward = truncation_reward # reward for exceeding max length

        """
        Observations
        
        A state is a list of tokens consisting of non-terminals and terminals in the leaf nodes of the partial parse tree
        """
        self.observation_space = gymnasium.spaces.MultiDiscrete([self.vocabulary_size] * max_len) # could use spaces.Sequence or tokenizers pkg

        """
        Actions

        An action is a single production rule applied to a single non-terminal in the current state
        """
        self.action_space = gymnasium.spaces.Discrete(self._num_rules * self.max_len) # could use spaces.Sequence but length needs to be equal to sequence across obs and actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        "Construct observation from environment state"

        return np.pad(np.array([self.vocabulary[token] for token in self.symbols]), (0, max(0, self.max_len - len(self.symbols))))

    def _get_info(self):
        "Obtain auxiliary info returned by `step` and `reset`"

        return {"action_mask": self.get_action_mask()}

    def reset(self, seed=None, options=None):
        "Initiate a new episode"

        super().reset(seed=seed)

        self.symbols = [self.start_symbol] # list of tokens starts with start symbol

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def get_action_mask(self) -> np.ndarray:
        "Return valid action mask for current state"
        # TODO: make more efficient by pre-computing some stuff or using np loops

        mask = np.array([0] * (self._num_rules * self.max_len), dtype=np.int8)
        for nt_idx, symbol in enumerate(self.symbols):
            if type(symbol) == lark.grammar.NonTerminal:
                for rule_idx, rule in enumerate(self.parser.rules):
                    if symbol == rule.origin:
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
    
    def is_valid(self, action: int) -> bool:
        "Test if an action is valid"

        nt_idx, rule_idx = self.decode_action(action)
        return (
            nt_idx < len(self.symbols) and 
            rule_idx < len(self.parser.rules) and 
            self.symbols[nt_idx] == self.parser.rules[rule_idx].origin
        )

    def step(self, action):
        
        nt_idx, rule_idx = self.decode_action(action)
        if self.is_valid(action): # if valid action, replace existing non-terminal with rule expansion
            self.symbols[nt_idx:nt_idx+1] = self.parser.rules[rule_idx].expansion
        else:
            print(f'Attempted invalid action {action} with NT idx {nt_idx} and rule idx {rule_idx}')

        terminated = all(symbol in self.terminals for symbol in self.symbols)
        truncated = len(self.symbols) > self.max_len
        if truncated:
            self.symbols = self.symbols[:self.max_len]

        if terminated: # get reward from external MDP by using finished program as policy
            program_text = ' '.join(str(self.parser.get_terminal(symbol.name).pattern) for symbol in self.symbols)
            reward = self.reward_fn(program_text, self.mdp_config)
        elif truncated: # partial program with len >= max_len; cannot be used as policy
            reward = self.truncation_reward
        else: # partial program; cannot be used as policy
            reward = 0.0 # sum(-1.0 for symbol in self.symbols if type(symbol) == lark.grammar.NonTerminal)
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):

        def nt_str(non_terminal):
            return f'_{non_terminal.name}_'
        
        def t_str(terminal):
            return f'{self.parser._terminals_dict[terminal.name].pattern.value}'

        def sym_str(symbol):
            if type(symbol) == lark.grammar.NonTerminal:
                return nt_str(symbol)
            elif type(symbol) == lark.grammar.Terminal:
                return t_str(symbol)
        
        print(' '.join([sym_str(symbol) for symbol in self.symbols]))

    def close(self):
        pass
