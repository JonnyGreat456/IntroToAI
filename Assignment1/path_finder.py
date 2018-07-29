import math
import sys

int numrow, numcol

class Node:
    def __init__(self, x, y, value, f = 0, g = 0, h = 0, parent = 0):
        self.x = x # row
        self.y = y # column
        self.value = value # 1 for unblocked, 0 for blocked
        self.f = f # path cost + heuristic
        self.g = g # path cost
        self.h = h # heuristic
        self.parent = parent
    def update_f():
        self.f = self.g + self.h
    def location():
        return self.x, self.y

def generateNeighbors(map, origin_node):
    """
        Args:
            map: 2D Array of 0 and 1 values indicating blocked (0) and unblocked (1) cells
            origin_node: Node from which to derive its neighbors
        Returns:
            List of Nodes which are neighbors (UP, DOWN, LEFT, and RIGHT) of the origin_node
    """

    neighbors = []
    x, y = origin_node.location()

    if(x == 0): # not done
        left = Node()
        if()
        neighbors.append(left)
        neighbors.append(right)

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

    open_list = []
    open_list.append(Node(initial_x, initial_y, map[initial_x][initial_y]))
    closed_list = []

    while(len(open_list) != 0):
        q = open_list[0]
        for node in open_list: # determine lowest f score in open_list
            if(node.f < q.f):
                q = node
            else:
                continue
        if(q.x == goal_x and q.y == goal_y):
            closed_list.append((q.x, q.y))
            return closed_list
        else:
            closed_list.append((q.x, q.y))
            # create a generateNeighbors(Node) function

def getargs():
    # sys.argv stores the command line args
    initial_x = sys.argv[0]
    initial_y = sys.argv[1]
    goal_x = sys.argv[2]
    goal_y = sys.argv[3]
    environment_file_path = sys.argv[4]
    output_file = sys.argv[5]