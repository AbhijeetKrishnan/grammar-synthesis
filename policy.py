import argparse
import random
from typing import Dict, List, Optional, Union

import grammar_synthesis
import gymnasium
import lark.grammar
from lark import Lark

symbol = Union[lark.grammar.NonTerminal, lark.grammar.Terminal]

class UniformRandomSampler:
    """
    McKenzie, B. (1997). Generating Strings at Random from a Context Free Grammar. https://doi.org/10/97
    """

    def _build_rule_map(self, parser: Lark) -> None:
        "Build map of non-terminals to expansion index as required by algorithm"
        
        nt_cnt = 0
        for rule in parser.rules:
            origin = rule.origin
            if origin not in self._nt_map:
                self._nt_map[origin] = nt_cnt
                self._production_list.append([])
                nt_cnt += 1
            nt_idx = self._nt_map[origin]
            self._production_list[nt_idx].append(rule.expansion)

    def _nt_idx(self, nt: lark.grammar.NonTerminal) -> int:
        "Return non-terminal index i in production rules of grammar"

        assert nt in self._nt_map
        return self._nt_map[nt]

    def _build_f_memo(self, n: int):
        f_memo_tmp = [[] for _ in range(n + 1)]
        for _n in range(n + 1):
            f_memo_tmp[_n] = [[] for _ in range(len(self._nt_map.keys()))]
        for _n in range(n + 1):
            for i in range(len(self._nt_map.keys())):
                f_memo_tmp[_n][i] = None
        self._f_memo = f_memo_tmp

    def _build_f_prime_memo(self, n: int):
        f_prime_memo_tmp = [[] for _ in range(n + 1)]
        for _n in range(n + 1):
            f_prime_memo_tmp[_n] = [[] for _ in range(len(self._nt_map.keys()))]
        for _n in range(n + 1):
            for i in range(len(self._nt_map.keys())):
                f_prime_memo_tmp[_n][i] = [[] for _ in range(len(self._production_list[i]))]
        for _n in range(n + 1):
            for i in range(len(self._nt_map.keys())):
                for j in range(len(self._production_list[i])):
                    f_prime_memo_tmp[_n][i][j] = [None for _ in range(len(self._production_list[i][j]))]
        self._f_prime_memo = f_prime_memo_tmp
    
    def __init__(self, parser: Lark, n: int):
        self._parser = parser

        self._nt_map: Dict[lark.grammar.NonTerminal, int] = {}
        self._production_list: List[List[symbol]] = []

        self._build_rule_map(parser)

        self._f_memo: List[List[Optional[List[int]]]] = []
        self._f_prime_memo: List[List[List[List[Optional[List[int]]]]]] = []

        self._build_f_memo(n)
        self._build_f_prime_memo(n)

    def _f(self, n: int, i: int) -> List[int]:
        "Return a list giving the number of strings of length n generated for each production N_i -> \alpha_(i,j)"

        if self._f_memo[n][i] is None:
            self._f_memo[n][i] = [sum(self._f_prime(n, i, j, 0)) for j in range(len(self._production_list[i]))]
        return self._f_memo[n][i]
    
    def _f_prime(self, n: int, i: int, j: int, k: int) -> List[int]:
        """
        Return the number of strings of length n generated by the final symbols
        x_(i,j,k) ... x_(i,j,t_(i,j)) for the RHS of the production \pi_(i,j): N_i -> x_(i,j,1) ... x_(i,j,t_(i,j))
        for each of the possible ways in which the n terminals can be split between the first symbol x_(i,j,k)
        and the remaining symbols

        n -> string length [0, n]
        i -> non-terminal index [0, ||N||)
        j -> production rule index for a given non-terminal i, s_i total production rules per non-terminal i
        k -> RHS symbol index for a given production rule (i, j)
        """

        if n < 0:
            return []
        if self._f_prime_memo[n][i][j][k] is None:
            if n == 0:
                self._f_prime_memo[n][i][j][k] = []
            elif type(self._production_list[i][j][k]) == lark.grammar.Terminal:
                if k + 1 == len(self._production_list[i][j]): # if last symbol of expansion
                    if n == 1:
                        self._f_prime_memo[n][i][j][k] = [1]
                    else:
                        self._f_prime_memo[n][i][j][k] = [0]
                else:
                    self._f_prime_memo[n][i][j][k] = [sum(self._f_prime(n - 1, i, j, k + 1))]
            elif k + 1 == len(self._production_list[i][j]):
                idx = self._nt_idx(self._production_list[i][j][k])
                self._f_prime_memo[n][i][j][k] = [sum(self._f(n, idx))]
            else:
                idx = self._nt_idx(self._production_list[i][j][k])
                self._f_prime_memo[n][i][j][k] = [sum(self._f(l, idx)) * sum(self._f_prime(n - l, i, j, k + 1)) for l in range(1, n - len(self._production_list[i][j]) + (k + 1) + 1)]
        # print('self._f_prime', n, i, j, k,self._f_prime_memo[n][i][j][k])
        return self._f_prime_memo[n][i][j][k]

    def _choose(self, l: List[int]) -> int:
        "Return an index i between [0, len(l)) at random with probability l[i] / sum(l)"

        return random.choices(range(len(l)), weights=l, k=1)[0]

    def _g(self, n: int, i: int) -> List[lark.grammar.Terminal]:
        "Generate a string of length n uniformly at random from a non-terminal N_i"

        if n < 0:
            return []
        r = self._choose(self._f(n, i))
        return self._g_prime(n, i, r, 0)

    def _g_prime(self, n: int, i: int, j: int, k: int) -> List[lark.grammar.Terminal]:
        """
        Generate a string of length n uniformly at random from among all the string derivable from the symbols
        x_(i,j,k) ... x_(i,j,t_(i,j)) taken from the RHS of the production N_i -> \alpha_(i,j)
        """

        if n < 0:
            return []
        if type(self._production_list[i][j][k]) == lark.grammar.Terminal:
            if k + 1 == len(self._production_list[i][j]):
                return [self._production_list[i][j][k]]
            else:
                return [self._production_list[i][j][k]] + self._g_prime(n - 1, i, j, k + 1)
        if k + 1 == len(self._production_list[i][j]):
            return self._g(n, self._nt_idx(self._production_list[i][j][k]))
        else:
            weights = self._f_prime(n, i, j, k)
            l = self._choose(weights) + 1
            return self._g(l, self._nt_idx(self._production_list[i][j][k])) + self._g_prime(n - l, i, j, k + 1)

    def generate_string(self, n: int):
        symbol_list = self._g(n, 0)
        program = ' '.join([self._parser._terminals_dict[terminal.name].pattern.value for terminal in symbol_list])
        return program

    def get_action(self, obs):
        pass

class RandomSampler:
    def __init__(self, env):
        self._action_space = env.action_space

    def get_action(self, obs, mask):
        return self._action_space.sample(mask=mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_episodes', type=int, default=5)
    parser.add_argument('-l', '--max_len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    env = gymnasium.make(
        'GrammarSynthesisEnv-v0', 
        grammar=open('grammar_synthesis/envs/assets/example.lark').read(), 
        start_symbol='s', 
        reward_fn=lambda program_text, mdp_config: len(program_text),
        max_len=args.max_len)
    
    num_episodes = args.num_episodes
    env.action_space.seed(args.seed)
    max_len = args.max_len

    uniform_agent = UniformRandomSampler(env.parser, max_len)
    random_agent = RandomSampler(env)

    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = random_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()

    program = uniform_agent.generate_string(5)
    print(program)