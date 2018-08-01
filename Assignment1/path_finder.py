import math
import sys

class Node:
    def __init__(self, x, y, g = 0, h = 0, parent = None):
        self.x = x # row
        self.y = y # column
        self.f = g + WEIGHT * h # path cost + heuristic
        self.g = g # path cost
        self.h = h # heuristic
        self.parent = parent
    def update_f(self, node):
        self.f = node.f
    def location(self):
        return (self.x, self.y)
########################################################################################
### global variables
########################################################################################
ROW_NUM = 0
COL_NUM = 0
START_NODE = None
GOAL_NODE = None
WEIGHT = 1
ZERO_HEURISTICS = 1
Output_File = None
MAP = []
PATH_COST = 0
PATH = []
#numExpansions = 0
#numSteps = 0

########################################################################################
### functions
########################################################################################

def generateNeighbors(origin_node):
    """
        Args:
            map: 2D Array of 0 and 1 values indicating blocked (0) and unblocked (1) cells
            origin_node: Node from which to derive its neighbors
        Returns:
            List of Nodes which are neighbors (UP, DOWN, LEFT, and RIGHT) of the origin_node
    """

    neighbors = []
    x = origin_node.location()[0]
    y = origin_node.location()[1]
    if(x - 1 < 0):
        pass
    else:
        if(MAP[x-1][y] == 1):#getting UP
            neighbors.append(Node(x-1,y, parent=origin_node))
    if(x + 1 >= len(MAP)):
        pass
    else:
        if(MAP[x+1][y] == 1):# getting DOWN
            neighbors.append(Node(x+1,y, parent=origin_node))
           
    if(y - 1 < 0):
        pass
    else:
        if(MAP[x][y-1] == 1):#getting LEFT
            neighbors.append(Node(x,y-1, parent=origin_node))
    if(y + 1 >= len(MAP[x])):
        pass
    else:
        if(MAP[x][y+1] == 1):# getting RIGHT
            neighbors.append(Node(x,y+1, parent=origin_node))
    return neighbors

def h(node): 
    """ function will compute the heuristic. It will compute the manhattan distance if zero_heuristic variable is one and
    and the zero heuristic if zero is zero """
    return ZERO_HEURISTICS * round( math.sqrt(((GOAL_NODE.x - node.x)**2)+((GOAL_NODE.y - node.y)**2)), 2)

def g(node):
    """ returns cost from the origin node to any node """
    #return round( math.sqrt(((node.x - origin_node.x)**2)+((node.y - origin_node.y)**2)), 2)
    return node.parent.g + 1

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
    rPath = []
    ptr = node
    pathCost = node.g
    while(ptr != None):
        rPath.append((ptr.x,ptr.y))
        ptr = ptr.parent
    #numSteps = len(rPath) - 1
    rPath.reverse()
    return pathCost, rPath
        
        
def PathFinder():
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
    global START_NODE, GOAL_NODE, PATH_COST, PATH

    open_list = []
    open_list.append(START_NODE)
    closed_list = []
    neighbors = []

    while(len(open_list) != 0):
        q = open_list.pop(min_f(open_list))
        if(q.x == GOAL_NODE.x and q.y == GOAL_NODE.y):
            PATH_COST, PATH = path_build(q)
            return PATH_COST, PATH
        if q not in closed_list:
            neighbors = generateNeighbors(q)
            for n in neighbors:
                n.g = g(n)
                n.h = h(n)
                n.f = n.g + (WEIGHT * n.h)
                n.parent = q
                if n not in open_list:
                    open_list.append(n)
                    #numExpansion += 1
                else:
                    found_n = open_list[open_list.index(n)]
                    if(n.f < found_n.f):
                        found_n.update_f(n)
       
# create a generateNeighbors(Node) function

def processMap(map_file_path):
    global ROW_NUM, COL_NUM, MAP
    map_file = open(map_file_path, "r")
    ROW_NUM = int(map_file.readline())
    COL_NUM = int(map_file.readline())

    for i in range(ROW_NUM):
        line = map_file.readline().split()
        num_list = []
        for j in range(COL_NUM):
            num_list.append(int(line[j]))
        MAP.append(num_list)

def getArgs():
    # retrieves command line arguments and stores them in global vars
    global START_NODE, GOAL_NODE, Output_File
    START_NODE = Node(int(sys.argv[1]), int(sys.argv[2]))
    GOAL_NODE = Node(int(sys.argv[3]), int(sys.argv[4]))
    enviroment_file_path = sys.argv[5]
    Output_File = sys.argv[6]

    processMap(enviroment_file_path)

def output_to_file():
    global Output_File, PATH_COST, PATH

    try:
        output_file = open(Output_File, "w+")
    except:
        print "Failed to open " + Output_File

    print PATH
    for node in PATH:
        try:
            output_file.write("%d %d\n" % (node[0], node[1]))
        except:
            print "Failed to write"
    
    output_file.close()


getArgs()
PathFinder()
output_to_file()
print "This finished first"