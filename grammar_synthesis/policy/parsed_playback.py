from collections.abc import Iterable

import parglare

# Ref.: https://stackoverflow.com/a/2158532
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        elif x is not None:
            yield x
            
class ParsedPlayback:
    "A policy that parses a program into a sequence of actions and plays them back"

    def __init__(self, env):
        self._grammar = env.grammar
        self._parser = parglare.Parser(self._grammar, build_tree=True)
        self._actions = None
        self._curr_idx = None

    def _visit(self, node, subresults, depth):
        s = None
        if node.is_nonterm():
            s = [node.production]
        if subresults:
            s += subresults
        return s

    def build_actions(self, program: str):
        parse_tree = self._parser.parse(program)
        result = parglare.visitor(parse_tree, parglare.trees.tree_node_iterator, self._visit)
        actions = list(flatten(result))

        self._actions = []
        self._curr_idx = 0
        symbol_list = [self._grammar.start_symbol]
        for production in actions:
            lhs = production.symbol
            rhs = production.rhs
            # replace first instance of lhs in symbol_list with rhs
            for i, symbol in enumerate(symbol_list):
                if symbol == lhs:
                    symbol_list = symbol_list[:i] + rhs + symbol_list[i+1:]
                    self._actions.append((i, production.prod_id - 1))
                    break

    def get_action(self, obs, mask=None):
        if self._curr_idx >= len(self._actions):
            self._curr_idx = 0
        action = self._actions[self._curr_idx]
        self._curr_idx += 1
        return action
