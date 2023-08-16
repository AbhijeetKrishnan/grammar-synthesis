import lark
from lark.parsers.earley_forest import ForestVisitor

class RuleSequencer(ForestVisitor):
    
    def __init__(self, single_visit=False):
        super().__init__(single_visit=single_visit)
        self.sequence = []
    
    def visit_packed_node_in(self, node):
        print('packed', node)
        return node.children

    def visit_symbol_node_in(self, node):
        # print('symbol', node)
        if type(node.s) == tuple:
            print('symbol', node.s)
        return node.children

class ParsedPlayback:
    "A policy that parses a program into a sequence of actions and plays them back"

    def __init__(self, parser: lark.Lark, program: str):
        self._parser = parser
        self._program = program
        self._visitor = RuleSequencer(single_visit=True)

    def _parse_program(self):
        sppf = self._parser.parse(self._program)
        self._visitor.visit(sppf)
        print(self._visitor.sequence)
        import code; code.interact(local=dict(globals(), **locals()))
        # print(sppf.pretty())

    def get_action(self, obs, mask=None):
        pass
        

if __name__ == '__main__':
    parser = lark.Lark(open('grammar_synthesis/envs/assets/example.lark').read(), start='s', parser='earley', ambiguity='forest')
    program = 'lool'
    playback_agent = ParsedPlayback(parser, program)
    playback_agent._parse_program()