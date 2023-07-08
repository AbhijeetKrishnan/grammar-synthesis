from typing import Tuple

import gymnasium
import lark
import lark.grammar
import numpy as np


class GrammarSynthesisEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": None}

    def __init__(self, grammar: str, start_symbol:str, reward_fn, max_len: int=200, render_mode=None, parser:str = 'earley'):
        self.parser = lark.Lark(grammar, parser=parser, start=start_symbol)
        self.start_symbol = self.parser.rules[0].origin
        self._num_rules = len(self.parser.rules)
        self.max_len = max_len
        self.terminals = [lark.grammar.Terminal(terminal_def.name) for terminal_def in self.parser.terminals]
        self.non_terminals = list({rule.origin for rule in self.parser.rules})
        self.vocabulary = {token: id for (token, id) in zip(self.terminals + self.non_terminals, range(len(self.terminals) + len(self.non_terminals)))}
        self.vocabulary_size = len(self.vocabulary)
        self.symbols = []
        self.reward_fn = reward_fn

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

        # print(self.symbols)
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
    
    def get_action_mask(self):
        "Return valid action mask for current state"
        # TODO: make more efficient by pre-computing some stuff or using np loops

        mask = np.array([0] * (self._num_rules * self.max_len), dtype=np.int8)
        for nt_idx, symbol in enumerate(self.symbols):
            if type(symbol) == lark.grammar.NonTerminal:
                for rule_idx, rule in enumerate(self.parser.rules):
                    if symbol == rule.origin:
                        mask[rule_idx * self.max_len + nt_idx] = 1 # if 0, action is masked (illegal), else not masked (legal)
        return mask
    
    def act_to_action(self, act: Tuple[int, int]):
        nt_idx, rule_idx = act
        action = rule_idx * self.max_len + nt_idx
        return action

    def step(self, action):
        rule_idx, nt_idx = action // self.max_len, action % self.max_len
        # print(nt_idx, rule_idx)
        # print('\n'.join([str(t) for t in enumerate(self.symbols)]))
        # print('\n'.join([str(t) for t in enumerate(self.parser.rules)]))
        # print('\n'.join([f'({idx}, {str(token)})' for token, idx in self.vocabulary.items()]))
        assert self.symbols[nt_idx] == self.parser.rules[rule_idx].origin

        self.symbols[nt_idx:nt_idx+1] = self.parser.rules[rule_idx].expansion

        terminated = all(symbol in self.terminals for symbol in self.symbols)
        truncated = len(self.symbols) > self.max_len
        if terminated:
            reward = self.reward_fn(self.symbols)
        else:
            reward = 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        print(' '.join([str(symbol.name) for symbol in self.symbols]))

    def close(self):
        pass
