import yaml

def load_graph(graph_path):
    with open(graph_path, "r") as f:
        graph_data = yaml.safe_load(f)
    
    graph = [[None]]*len(graph_data)
    for node, next in graph_data.items():
        if next != "None":
            if type(next) == list:
                graph[node] = next
            elif type(next) == int:
                graph[node] = [next]
    
    return graph

def calc_route(graph, start, goal):
    if goal in graph[start]:
        

path = "./images/test/graph.yaml"
graph = load_graph(path)

print(graph)