import random
from typing import Dict, List, Optional, Union

import parglare
import parglare.grammar


class UniformRandomSampler:
    """
    A grammar synthesis policy that randomly samples from available actions by uniformly weighting the number of strings
    that can be produced.

    Implements the algorithm due to McKenzie, B. (1997). Generating Strings at Random from a Context Free 
    Grammar. https://doi.org/10/97
    """

    def __init__(self, env, n: int, seed: Optional[int] = None):
        self._grammar = env.grammar
        self._parser = parglare.Parser(self._grammar, build_tree=True)
        random.seed(seed)

        self._num_nonterminals = len(self._grammar.nonterminals) - 1 # parglare adds an extra non-terminal S' for the augmented rule S' -> [start_symbol]
        self._nonterminals: List[parglare.grammar.NonTerminal] = [value for key, value in self._grammar.nonterminals.items() if key != "S'"]
        self._nt_idx = {nonterminal: idx for nonterminal, idx in zip(self._nonterminals, range(self._num_nonterminals))}
        self._production_list: List[List[parglare.grammar.ProductionRHS]] = []

        # self._build_rule_map()

        self._f_memo: List[List[Optional[List[int]]]] = []
        self._f_prime_memo: List[List[List[List[Optional[List[int]]]]]] = []

        self._build_f_memo(n)
        self._build_f_prime_memo(n)

        self.actions = []
    
    # def _build_rule_map(self) -> None:
    #     "Build map of non-terminals to expansion index as required by the algorithm"
        
    #     nt_cnt = 0
    #     for rule in self._grammar.productions[1:]:
    #         origin = rule.symbol
    #         if origin not in self._nt_map:
    #             self._nt_map[origin] = nt_cnt
    #             self._production_list.append([])
    #             nt_cnt += 1
    #         nt_idx = self._nt_map[origin]
    #         self._production_list[nt_idx].append(rule.rhs)

    # def _nt_idx(self, nt: parglare.grammar.NonTerminal) -> int:
    #     "Return non-terminal index i in the production rules of the grammar"

    #     assert nt in self._nt_map
    #     return self._nt_map[nt]

    def _build_f_memo(self, n: int):
        f_memo_tmp = [[] for _ in range(n + 1)]
        for _n in range(n + 1):
            f_memo_tmp[_n] = [[] for _ in range(self._num_nonterminals)]
        for _n in range(n + 1):
            for i in range(self._num_nonterminals):
                f_memo_tmp[_n][i] = None
        self._f_memo = f_memo_tmp

    def _build_f_prime_memo(self, n: int):
        f_prime_memo_tmp = [[] for _ in range(n + 1)]
        for _n in range(n + 1):
            f_prime_memo_tmp[_n] = [[] for _ in range(self._num_nonterminals)]
        for _n in range(n + 1):
            for i in range(self._num_nonterminals):
                f_prime_memo_tmp[_n][i] = [[] for _ in range(len(self._nonterminals[i].productions))]
        for _n in range(n + 1):
            for i in range(self._num_nonterminals):
                for j in range(len(self._nonterminals[i].productions)):
                    f_prime_memo_tmp[_n][i][j] = [None for _ in range(len(self._nonterminals[i].productions[j].rhs))]
        self._f_prime_memo = f_prime_memo_tmp

    def _f(self, n: int, i: int) -> List[int]:
        "Return a list giving the number of strings of length n generated for each production N_i -> \alpha_(i,j)"

        if self._f_memo[n][i] is None:
            self._f_memo[n][i] = [sum(self._f_prime(n, i, j, 0)) for j in range(len(self._nonterminals[i].productions))]
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
            elif type(self._nonterminals[i].productions[j].rhs[k]) == parglare.grammar.Terminal:
                if k + 1 == len(self._nonterminals[i].productions[j].rhs): # if last symbol of expansion
                    if n == 1:
                        self._f_prime_memo[n][i][j][k] = [1]
                    else:
                        self._f_prime_memo[n][i][j][k] = [0]
                else:
                    self._f_prime_memo[n][i][j][k] = [sum(self._f_prime(n - 1, i, j, k + 1))]
            elif k + 1 == len(self._nonterminals[i].productions[j].rhs):
                idx = self._nt_idx[self._nonterminals[i].productions[j].rhs[k]]
                self._f_prime_memo[n][i][j][k] = [sum(self._f(n, idx))]
            else:
                idx = self._nt_idx[self._nonterminals[i].productions[j].rhs[k]]
                self._f_prime_memo[n][i][j][k] = [sum(self._f(l, idx)) * sum(self._f_prime(n - l, i, j, k + 1)) for l in range(1, n - len(self._nonterminals[i].productions[j].rhs) + (k + 1) + 1)]
        # print('self._f_prime', n, i, j, k,self._f_prime_memo[n][i][j][k])
        return self._f_prime_memo[n][i][j][k]

    def _choose(self, l: List[int]) -> int:
        "Return an index i between [0, len(l)) at random with probability l[i] / sum(l)"

        return random.choices(range(len(l)), weights=l, k=1)[0]

    def _g(self, n: int, i: int) -> List[parglare.grammar.Terminal]:
        "Generate a string of length n uniformly at random from a non-terminal N_i"

        if n < 0:
            return []
        r = self._choose(self._f(n, i))
        return self._g_prime(n, i, r, 0)

    def _g_prime(self, n: int, i: int, j: int, k: int) -> List[parglare.grammar.Terminal]:
        """
        Generate a string of length n uniformly at random from among all the strings derivable from the symbols
        x_(i,j,k) ... x_(i,j,t_(i,j)) taken from the RHS of the production N_i -> \alpha_(i,j)
        """

        if n < 0:
            return []
        if type(self._nonterminals[i].productions[j].rhs[k]) == parglare.grammar.Terminal:
            if k + 1 == len(self._nonterminals[i].productions[j].rhs):
                # self.actions.append(self._nonterminals[i].productions[j])
                return [self._nonterminals[i].productions[j].rhs[k]]
            else:
                # self.actions.append(self._nonterminals[i].productions[j])
                return [self._nonterminals[i].productions[j].rhs[k]] + self._g_prime(n - 1, i, j, k + 1)
        if k + 1 == len(self._nonterminals[i].productions[j].rhs):
            return self._g(n, self._nt_idx[self._nonterminals[i].productions[j].rhs[k]])
        else:
            weights = self._f_prime(n, i, j, k)
            l = self._choose(weights) + 1
            return self._g(l, self._nt_idx[self._nonterminals[i].productions[j].rhs[k]]) + self._g_prime(n - l, i, j, k + 1)

    def generate_string(self, n: int):
        self.actions = []
        symbol_list = self._g(n, 0)
        program = ' '.join(str(symbol.name) for symbol in symbol_list) # TODO: duplicated from synthesis_env
        return program

    def get_action(self, obs, mask=None):
        pass # TODO:
