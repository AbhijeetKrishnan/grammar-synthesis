import parglare

class ParsedPlayback:
    "A policy that parses a program into a sequence of actions and plays them back"

    def __init__(self, env):
        self._grammar = env.grammar
        self._parser = parglare.Parser(self._grammar, build_tree=True)
        self._actions = None
        self._curr_idx = None

    def _visit(self, node, subresults, depth):
        if node.is_nonterm():
            self._actions.append(node.production)
            return subresults

    def build_actions(self, program: str):
        self._actions = []
        self._curr_idx = 0
        parse_tree = self._parser.parse(program)
        parglare.visitor(parse_tree, parglare.trees.tree_node_iterator, self._visit)
        self._actions.reverse()
        actions = self._actions.copy()
        self._actions = []
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
