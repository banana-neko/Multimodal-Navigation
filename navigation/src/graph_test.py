from utils import load_nodes

x = []

if not x:
    print("x is empty.")
    
for a in x:
    print(a)
    if a == None:
        print(f"a is None.")

print("finish")

nodes = load_nodes("../maps/0/graph/nodes.yaml")
print(nodes)