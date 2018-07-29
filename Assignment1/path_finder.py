import math

class Node:
    def __init__(self, x, y, f = 0, g = 0, h = 0, parent = 0):
        self.x = x # row
        self.y = y # column
        self.f = f # path cost + heuristic
        self.g = g # path cost
        self.h = h # heuristic
        self.parent = parent
    def update_f(self, node):
        self.f = node.f
    def location(self):
        return self.x, self.y
########################################################################################
### global variables
########################################################################################
START_NODE = None
GOAL_NODE = None
WEIGHT = 1
ZERO_HEURISTICS = 1
numExpansions = 0
numSteps = 0

########################################################################################
### functions
########################################################################################

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
    if(x - 1 < 0):
        pass
    else:
        if(map[x-1][y] == 1):#getting UP
            neighbors.append(Node(x-1,y,1))
    if(x + 1 >= len(map)):
        pass
    else:
        if(map[x+1][y] == 1):# getting DOWN
            neighbors.append(Node(x+1,y,1))
           
    if(y - 1 < 0):
        pass
    else:
        if(map[x][y-1] == 1):#getting LEFT
            neighbors.append(Node(x,y-1,1))
    if(y + 1 >= len(map[x])):
        pass
    else:
        if(map[x][y+1] == 1):# getting RIGHT
            neighbors.append(Node(x,y+1,1))
    return neighbors
def h(origin_node, goal_node): 
    """ function will compute the heuristic. It will compute the manhattan distance if zero_heuristic variable is one and
    and the zero heuristic if zero is zero """
    return ZERO_HEURISTIC * round( math.sqrt(((goal_node.x - origin_node.x)**2)+((goal_node.y - origin_node.y)**2)), 2)

def g(origin_node, node):
    """ returns cost from the origin node to any node """
    return round( math.sqrt(((node.x - origin_node.x)**2)+((node.y - origin_node.y)**2)), 2)
def min_f(queue):
    """ finds the index of the node with the lowest f value in open_list """
    min_value_index = 0
    if(len(queue) == 1):
        return 0
    for i in range(1,len(queue)):
        if(queue[min_value_index].f <= queue[i].f):
            pass
        else:
            min_value_index = i
    return min_value_index
def path_build(node):
    """ This function will iterate through a path through a list of its parents """
    rPath = list()
    pathCost = 0
    ptr = node
    while(ptr != None):
        rPath.append((ptr.x,ptr.y))
        pathCost = pathCost + ptr.g
        ptr = ptr.parent
    numSteps = len(rPath) - 1
    return PathCost, rPath.reverse()
        
    
        
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

    #pathExists = True # boolean flag for whether or not path exists, assumed True until determined False don't think we need this

    open_list = []
    path = []
    pathCost = 0
    GOAL_NODE = Node(goal_x,goal_y)
    START_NODE = Node(initial_x, initial_y)
    open_list.append(START_NODE)
    closed_list = []
    neighbors = []

    while(len(open_list) != 0):
        q = open_list.pop(min_f(open_list))
        if(q.x == GOAL_NODE.x AND q.y == GOAL_NODE.y):
            pathCost, path = path_build(GOAL_NODE)
            return pathCost, path
        if q not in closed_list:
            neighbors = generateNeighbors(map, q)
            for n in neighbors:
                n.g = g(q,n)
                n.h = h(n,GOAL_NODE)
                n.f = n.g + (WEIGHT * n.h)
                n.parent = q
                if n not in open_list:
                    open_list.append(n)
                    numExpansions = numExpansions + 1
                else:
                    found_n = open_list[open_list.index(n)]
                    if(n.f < found_n.f):
                        found_n.update_f(n)
       
# create a generateNeighbors(Node) function
