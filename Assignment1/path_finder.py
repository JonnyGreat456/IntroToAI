import math
import os
import sys

class Node:
    def __init__(self, x, y, value):
        self.x = x # row
        self.y = y # column
        self.value = value # 1 for unblocked, 0 for blocked
        self.f = 0 # path cost + heuristic
        self.g = 0 # path cost
        self.h = 0 # heuristic
        self.parent = 0

def generateMap(map_filename):
    map_file = open(map_filename, "r")
    ROW_NUM = int(map_file.readline())
    COL_NUM = int(map_file.readline())
    MAP = []

    for i in range(ROW_NUM):
        line = map_file.readline().split()
        num_list = []
        for j in range(COL_NUM):
            num_list.append(int(line[j]))
        MAP.append(num_list)

    return MAP

def generateNeighbors(map, origin_node):
    """
        Args:
            map: 2D Array of 0 and 1 values indicating blocked (0) and unblocked (1) cells
            origin_node: Node from which to derive its neighbors
        Returns:
            List of Nodes which are neighbors (UP, DOWN, LEFT, and RIGHT) of the origin_node
    """

    neighbors = []
    x = origin_node.x
    y = origin_node.y

    if(x - 1 < 0):
        pass
    else:
        if(map[x - 1][y] == 1):
            up = Node(x - 1, y, map[x - 1][y])
            neighbors.append(up)

    if(x + 1 >= len(map)):
        pass
    else:
        if(map[x + 1][y] == 1):
            down = Node(x + 1, y, map[x + 1][y])
            neighbors.append(down)

    if(y - 1 < 0):
        pass
    else:
        if(map[x][y - 1] == 1):
            left = Node(x, y - 1, map[x][y - 1])
            neighbors.append(left)

    if(y + 1 >= len(map[x])):
        pass
    else:
        if(map[x][y + 1] == 1):
            right = Node(x, y + 1, map[x][y + 1])
            neighbors.append(right)

    return neighbors

def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    i = 0
    j = 0
    k = l

    while((i < n1) and (j < n2)):
        if(L[i].f <= R[j].f):
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while(i < n1):
        arr[k] = L[i]
        i += 1
        k += 1

    while(j < n2):
        arr[k] = R[j]
        j += 1
        k += 1

def MergeSort(arr, l, r): # implementation of merge sort for sorting open list
    if(l < r):
        m = (l + (r - 1)) / 2
        MergeSort(arr, l, m)
        MergeSort(arr, m + 1, r)
        merge(arr, l, m, r)

def sort_node_list(node_list): # merge sort wrapper function
    MergeSort(node_list, 0, len(node_list) - 1)

def isNodeInList(node_list, a_node):
    x = a_node.x
    y = a_node.y
    for node in node_list:
        x2 = node.x
        y2 = node.y
        if(x == x2 and y == y2):
            return True
        else:
            continue
    return False

def isNodeInList2(tuple_list, a_node):
    x = a_node.x
    y = a_node.y
    for t in tuple_list:
        x2 = t[0]
        y2 = t[1]
        if(x == x2 and y == y2):
            return True
        else:
            continue
    return False

def g(origin_node, node):
    """ returns cost from the origin node to any node """
    return round(math.sqrt(((node.x - origin_node.x)**2) + ((node.y - origin_node.y)**2)), 2)

def h(origin_node, goal_node):
    """ function will compute the heuristic. """
    return round((math.fabs(origin_node.x - goal_node.x) + math.fabs(origin_node.y - goal_node.y)), 2)

def find(queue, targetNode):
    """ looks for a node in a queue and returns its index or -1 if node is not found """
    x = targetNode.x
    y = targetNode.y
    for i in range(0,len(queue)):
        x2 = queue[i].x
        y2 = queue[i].y
        if(x == x2 and y == y2):
            return i
    return -1

def PathFinder(map, initial_x, initial_y, goal_x, goal_y):
    """
        Args:
            map: 2D Array of 0 and 1 values indicating blocked (0) and unblocked (1) cells
            initial_x: integer value denoting the starting row
            initial_y: integer value denoting the starting column
            goal_x: integer value denoting the goal row
            goal_y: integer value denoting the goal column
        Returns:
            List of 2-tuples indicating a path from the starting state to the goal state
                i.e. path from (0,0) -> (2,2): ((0,0), (0,1), (1,1), (1,2), (2,2))
    """

    pathExists = True # boolean flag for whether or not path exists, assumed True until determined False
    numExpansions = 0

    open_list = []
    open_list.append(Node(initial_x, initial_y, map[initial_x][initial_y]))
    closed_list = []

    while(len(open_list) != 0):
        sort_node_list(open_list) # determine lowest f score in open_list by sorting on f score
        q = open_list.pop(0)
        if(q.x == goal_x and q.y == goal_y):
            closed_list.append((q.x, q.y))
            print('Number of Expansions: %d' % numExpansions)
            return closed_list
        else:
            closed_list.append((q.x, q.y))
            neighbors = generateNeighbors(map, q)

            for n in neighbors:
                if(isNodeInList2(closed_list, n)):
                    continue
                else:
                    numExpansions += 1
                    n.g = q.g + g(q, n)
                    n.h = h(n, Node(goal_x, goal_y, map[goal_x][goal_y]))
                    n.f = n.g + n.h
                    n.parent = q

                    if not isNodeInList(open_list, n):
                        open_list.append(n)
                    else:
                        found_n = open_list[find(open_list, n)]
                        if(n.f < found_n.f):
                            found_n.f = n.f


# Main Execution Test
prefix = os.environ['PRACSYS_PATH']
map = generateMap(prefix + '/prx_core/launches/maze')
path = PathFinder(map, 0, 0, 3, 8)
print(path)
