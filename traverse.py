
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}


def dfs(graph, node, visited=None):

    if visited is None:
        visited = set()

    visited.add(node)
    print(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)



def bfs(graph, start):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            print(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)   

print("DFS:")
dfs(graph, 'A')
print("\nBFS:")
bfs(graph, 'A')
