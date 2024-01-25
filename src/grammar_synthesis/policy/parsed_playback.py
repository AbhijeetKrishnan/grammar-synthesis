from typing import Any, Generator, List, Tuple, TypeVar

import numpy as np
import parglare
import parglare.grammar
import parglare.trees
from numpy.typing import NDArray

from grammar_synthesis.envs.synthesis_env import GrammarSynthesisEnv

T = TypeVar("T")


# Ref.: https://stackoverflow.com/a/2158532
def flattenList(xs: List[List[Any] | Any | None]) -> Generator[Any, None, None]:
    for x in xs:
        if isinstance(x, List):
            yield from flattenList(x)
        elif x is not None:
            yield x


class ParsedPlayback:
    "A policy that parses a program into a sequence of actions and plays them back"

    def __init__(self, env: GrammarSynthesisEnv) -> None:
        self._env = env
        self._grammar = env.grammar
        self._parser = parglare.Parser(self._grammar, build_tree=True)

        self._actions: List[np.uint8] = []
        self._curr_idx: int | None = None

    def _visit(
        self,
        node: parglare.trees.Node,
        subresults: List[parglare.grammar.Production] | None,
        depth: int,
    ) -> List[parglare.grammar.Production] | None:
        if node.is_nonterm() and subresults is not None:
            s = [node.production] + subresults
        elif node.is_nonterm():
            s = [node.production]
        elif subresults is not None:
            s = subresults
        else:
            s = None
        return s

    def build_actions(self, program: str) -> None:
        parse_tree = self._parser.parse(program)
        result = parglare.visitor(
            parse_tree, parglare.trees.tree_node_iterator, self._visit
        )
        actions: List[parglare.grammar.Production] = list(flattenList(result))

        self._actions = []
        self._curr_idx = 0
        symbol_list = [self._grammar.start_symbol]
        for production in actions:
            lhs = production.symbol
            rhs = production.rhs
            # replace left-most instance of lhs in symbol_list with rhs
            for i, symbol in enumerate(symbol_list):
                if symbol == lhs:
                    symbol_list = symbol_list[:i] + rhs + symbol_list[i + 1 :]
                    self._actions.append(production.prod_id - 1)
                    break

    def get_action(
        self,
        obs: NDArray[np.uintc],
        mask: np.ndarray[Tuple[int], np.dtype[np.int8]] | None = None,
    ) -> np.ulonglong:
        if self._curr_idx is None or self._curr_idx >= len(self._actions):
            self._curr_idx = 0
        decoded_action = self._actions[self._curr_idx]
        self._curr_idx += 1
        return decoded_action
