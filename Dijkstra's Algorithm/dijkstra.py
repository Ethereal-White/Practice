import numpy as np

'''drawing picture'''
m = 1000  # ∞
length = 16
startPointIndex = 0  # set the start point
graph = np.array([[0, 1, m, m, m, m, m, 9, m, m, m, m, m, m, m, m],  # B
                  [1, 0, 4, m, m, m, 10, m, m, m, m, m, m, m, m, m],  # C
                  [m, 4, 0, 1, m, 4, m, m, m, m, m, m, m, m, m, m],  # D
                  [m, m, 1, 0, 2, m, m, m, m, m, m, m, m, m, m, m],  # E
                  [m, m, m, 2, 0, 1, m, m, m, m, m, 15, m, m, m, m],  # F
                  [m, m, 4, m, 1, 0, 3, m, m, m, 10, m, m, m, m, m],  # G
                  [m, 10, m, m, m, 3, 0, 1, m, 6, m, m, m, m, m, m],  # H
                  [9, m, m, m, m, m, 1, 0, 3, m, m, m, m, m, m, m],  # I
                  [m, m, m, m, m, m, m, 3, 0, 1, m, m, m, m, 7, m],  # J
                  [m, m, m, m, m, m, 6, m, 1, 0, 2, m, m, 6, m, m],  # K
                  [m, m, m, m, m, 10, m, m, m, 2, 0, 1, 2, m, m, m],  # L
                  [m, m, m, m, 15, m, m, m, m, m, 1, 0, m, m, m, 1],  # M
                  [m, m, m, m, m, m, m, m, m, m, 2, m, 0, 7, m, 3],  # N
                  [m, m, m, m, m, m, m, m, m, 6, m, m, 7, 0, 1, m],  # O
                  [m, m, m, m, m, m, m, m, 7, m, m, m, m, 1, 0, m],  # P
                  [m, m, m, m, m, m, m, m, m, m, m, 1, 3, m, m, 0]])  # A

'''setting'''
s = [False] * length  # S set
s[startPointIndex] = True
dis = [i for i in graph[0]]  # distance to the start point

'''finding the min route'''
for i in range(length):
    tmpDis = m
    tmpIndex = 0
    for j in range(length):
        if dis[j] < tmpDis and not s[j]:
            tmpDis = dis[j]
            tmpIndex = j

    s[tmpIndex] = True  # add the point to set S

    for j in range(length):
        if dis[j] > dis[tmpIndex] + graph[tmpIndex][j]:
            dis[j] = dis[tmpIndex] + graph[tmpIndex][j]

print(dis)

# graph = np.array([[0, 4, 6, 6, m, m, m],
#                   [m, 0, 1, m, 7, m, m],
#                   [m, m, 0, m, 6, 4, m],
#                   [m, m, 2, 0, m, 5, m],
#                   [m, m, m, m, 0, m, 6],
#                   [m, m, m, m, 1, 0, 8],
#                   [m, m, m, m, m, m, m]])
