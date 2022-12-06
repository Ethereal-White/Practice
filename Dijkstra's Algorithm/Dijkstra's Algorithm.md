# Dijkstra-s-Algorithm

![0020a162f0cb8a5546686661af10eb9](https://user-images.githubusercontent.com/67407370/180425713-84b35754-be36-4ad4-8ac5-8f6f772db61b.png)

The question is: To find the shortest path between A and B

While I wrote the code of Dijkstra Algorithm which is in the python file in this directory, I also use ChatGPT, an AI, to define a function of Dijkstra algorithm. Here is the code. 

```python
INF = float("inf")

def Dijkstra(graph, start, end):
    # Initialize the distances to each vertex to infinity
    n = len(graph)
    dist = [INF] * n
    prev = [None] * n

    # Set the distance from the starting vertex to itself to 0
    dist[start] = 0

    # Create a set of unvisited vertices
    unvisited = set(range(n))

    # While there are still unvisited vertices
    while unvisited:
        # Find the unvisited vertex with the smallest distance from the starting vertex
        u = min(unvisited, key=lambda v: dist[v])

        # If the destination vertex has been reached, stop
        if u == end:
            break

        # Remove the vertex from the set of unvisited vertices
        unvisited.remove(u)

        # Update the distances to all the other vertices that can be reached from u
        for v in unvisited:
            alt = dist[u] + graph[u][v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    return dist, prev
```
```python
INF = float("inf")

def Dijkstra(graph, start):
    # Initialize the distances to each vertex to infinity
    n = len(graph)
    dist = [INF] * n
    prev = [None] * n
    paths = [[] for _ in range(n)]

    # Set the distance from the starting vertex to itself to 0
    dist[start] = 0

    # Create a set of unvisited vertices
    unvisited = set(range(n))

    # While there are still unvisited vertices
    while unvisited:
        # Find the unvisited vertex with the smallest distance from the starting vertex
        u = min(unvisited, key=lambda v: dist[v])

        # Remove the vertex from the set of unvisited vertices
        unvisited.remove(u)

        # Update the distances to all the other vertices that can be reached from u
        for v in unvisited:
            alt = dist[u] + graph[u][v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                paths[v] = paths[u] + [v]

    return dist, paths
```
