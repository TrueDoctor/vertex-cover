"""
    usage: python3 validator.py path/to/graph path/to/vertex/cover
"""

import sys

graph = sys.argv[1]
graph_file = open(graph, "r")

sol = sys.argv[2]
sol_file = open(sol, "r")

n, m = map(int, graph_file.readline().split())

lines = graph_file.readlines()
edges = list(map(int, l.split()) for l in lines[1:])

lines = sol_file.readlines();
vc_size = int(lines[0])
vc = set(int(s) for s in lines[1:])

if vc_size != len(vc):
    print("Printed size does not match")
    quit()

for v in vc:
    if v < 1 or v > n:
        print("ID {} not valid".format(v))

for a, b in edges:
    if a not in vc and b not in vc:
        print("Edge {} {} is not covered".format(a, b))
        quit()

print("Valid vertex cover")