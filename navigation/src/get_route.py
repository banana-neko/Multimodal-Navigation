import os
from utils import load_graph, load_nodes
from const import MAP_PATH
import heapq


map_id = "0"
map_path = os.path.join(MAP_PATH, map_id)
nodes = load_nodes(os.path.join(map_path, "graph", "nodes.yaml"))
graph = load_graph(os.path.join(map_path, "graph", "graph.yaml"))


def chack_reach(start, goal, nodes):
    def rec(start, goal, nodes, visited):
        if start in visited:
            return False
        visited.append(start)

        if goal in nodes[start]: #接続先にゴールがある
            return True
        if not nodes[start]: #終端ノード
            return False
        
        result = False
        for con_node in nodes[start]:
            if rec(con_node, goal, nodes, visited):
                result = True
        
        return result

    return rec(start, goal, nodes, visited=[])


def dijkstra(start, goal, graph):
    nodes = [float('inf')] * len(graph)
    nodes[0] = 0

    min_path = []
    heapq.heappush(min_path, [0, [0]]) # [cost, min_path]

    while len(min_path) > 0:
        _, min_point = heapq.heappop(min_path)
        last = min_point[-1]
        if last == goal:
            return min_point
        
        for factor in graph[last].items():
            t = 0

print(chack_reach(0, 7, nodes))