import heapq
from typing import Dict, List

def a_star(osmid2geo_3310, graph: Dict[str, Dict[str, float]], 
           start: str, end: str) -> List[str]:
    distances = {}

    # Heuristic function for estimating the distance between two nodes
    def h(node):
        if (node, end) not in distances:
            distances[(node, end)] = distances[(end, node)] = osmid2geo_3310[node].distance(osmid2geo_3310[end])
        return distances[(node, end)]
    
    # Initialize distance and previous node dictionaries
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = h(start)
    prev = {node: None for node in graph}
    
    # Initialize heap with start node and its f score
    heap = [(f_score[start], start)]
    
    while heap:
        # Pop the node with the smallest f score from the heap
        (f, curr_node) = heapq.heappop(heap)
        
        # If we have reached the end node, return the shortest path
        if curr_node == end:
            path = []
            while curr_node is not None:
                path.append(curr_node)
                curr_node = prev[curr_node]
                
            return path[::-1]
        
        # Otherwise, update the f and g scores of all adjacent nodes
        for neighbor, weight in graph[curr_node].items():
            # Check if there is an edge between the current node and the neighbor
            if neighbor not in g_score:
                continue
                
            new_g_score = g_score[curr_node] + weight
            if new_g_score < g_score[neighbor]:
                g_score[neighbor] = new_g_score
                f_score[neighbor] = new_g_score + h(neighbor)
                prev[neighbor] = curr_node
                heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    # If we get here, there is no path from start to end
    return None
