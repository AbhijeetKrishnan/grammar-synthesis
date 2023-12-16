LOL = """
s: s s | "l" a "l";

a: "o" a | "o";
"""

# Ref.: https://github.com/igordejanovic/parglare/blob/master/examples/json/json.pg
JSON = """
value: FALSE | TRUE | NULL | object | array | number | string;
object: "{" member*[COMMA] "}";
member: string ":" value;
array: "[" value*[COMMA] "]";

terminals
FALSE: 'false';
TRUE: 'true';
NULL: 'null';
COMMA: ',';
number: /-?\\d+(\\.\\d+)?(e|E[-+]?\\d+)?/;
string: /"((\\\\")|[^"])*"/;
"""
