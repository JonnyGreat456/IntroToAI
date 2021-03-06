import math
import argparse
import sys

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
        return (self.x, self.y)
    '''def traverse(self):
        node = self # start from the head node
        while node != None:
            if node == None:
                continue
            print node.location() # access the node value
            node = node.parent # move on to the next node'''
            
########################################################################################
### global variables
########################################################################################
ROW_NUM = 0
COL_NUM = 0
START_NODE = None
GOAL_NODE = None
WEIGHT = 1
DENSITY = 10 #please update for report purposes
ZERO_HEURISTIC = 1
Output_File = "/home/postalmist/pathlist/path.txt" # specify your own path
Report_File = "/home/postalmist/pathlist/report.txt" # specify your own path
MAP = []
enviroment_file_path = None
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
    x = origin_node.location()[0]
    y = origin_node.location()[1]
    if(x - 1 < 0):
        pass
    else:
        if(map[x-1][y] == 1):#getting UP
            neighbors.append(Node(x-1,y))
    if(x + 1 >= len(map)):
        pass
    else:
        if(map[x+1][y] == 1):# getting DOWN
            neighbors.append(Node(x+1,y))
           
    if(y - 1 < 0):
        pass
    else:
        if(map[x][y-1] == 1):#getting LEFT
            neighbors.append(Node(x,y-1))
    if(y + 1 >= len(map[x])):
        pass
    else:
        if(map[x][y+1] == 1):# getting RIGHT
            neighbors.append(Node(x,y+1))
    return neighbors
def h(origin_node, goal_node): 
    """ function will compute the heuristic. It will compute the manhattan distance if zero_heuristic variable is one and
    and the zero heuristic if zero is zero """
    global ZERO_HEURISTIC
    #return ZERO_HEURISTIC * round( math.sqrt(((goal_node.x - origin_node.x)**2)+((goal_node.y - origin_node.y)**2)), 2)
    return ZERO_HEURISTIC * round((math.fabs(origin_node.x - goal_node.x) + math.fabs(origin_node.y - goal_node.y)),2)

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
def find(queue, targetNode):
    """ looks for a node in a queue and returns its index or -1 if node is not found """
    for i in range(0,len(queue)):
        if(targetNode.location() == queue[i].location()):
            return i
    return -1
    
def path_build(node):
    """ This function will iterate through a path through a list of its parents """
    global numSteps
    rPath = list()
    cost =  node.g
    
    ptr = node
    
    while ptr != None:
        rPath.append(ptr.location())
        #print ptr
        ptr = ptr.parent
    numSteps = len(rPath) - 1
    rPath.reverse()
    
    #node.traverse()
    return cost, rPath
        
    
        
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

    global WEIGHT,START_NODE,GOAL_NODE,numExpansions
    open_list = []
    path = []
    pathCost = 0
    goal_node = Node(goal_x,goal_y,0,0,)
    start_node = Node(initial_x, initial_y,0,0,0,None)
    start_node.h = h(start_node,goal_node)
    open_list.append(start_node)
    closed_list = []
    neighbors = []

    while(len(open_list) != 0):
        q = open_list.pop(min_f(open_list))
        if(q.x == goal_node.x and q.y == goal_node.y):
            
            pathCost, path = path_build(q)
            return pathCost, path
        closed_list.append(q)
        numExpansions = numExpansions + 1
        neighbors = generateNeighbors(map, q)
        for n in neighbors:
            n.g = q.g + g(q,n)
            n.h = h(n,goal_node)
            #print WEIGHT
            n.f = n.g + (WEIGHT * n.h)
            n.parent = q
            if(find(closed_list,n) > -1):#has already been explored or expanded
                pass
            else:
                if(find(open_list,n) == -1): #not found in the openList
                    open_list.append(n)
                    #numExpansions = numExpansions + 1
                else:
                    found_n = open_list[find(open_list,n)]
                    if(n.f < found_n.f):
                        found_n.update_f(n)
                        

def processMap(map_file_path):
    """ reads map from text file and puts in an array"""
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

"""def getArgs():
    # retrieves command line arguments and stores them in global vars
    global START_NODE, GOAL_NODE, Output_File
    START_NODE = Node(sys.argv[0], sys.argv[1])
    GOAL_NODE = Node(sys.argv[2], sys.argv[3])
    enviroment_file_path = sys.argv[4]
    Output_File = sys.argv[5]

    processMap(enviroment_file_path)
 """

def output_to_file(path,pc):
    ''' outputs the path in a text file and report of the code to a report file'''
    global Output_File, Report_File, WEIGHT, DENSITY, ZERO_HEURISTIC,ROW_NUM,COL_NUM,numExpansions,numSteps

    output_file = open(Output_File, "w+")

    for node in path:
        output_file.write("%d %d\n" % (node[0], node[1]))
        
    output_file.close()
    
    ofile = open(Report_File, "a+")
    
    ofile.write ("Report for %d by %d map with %d%% density, weight = %d, and heuristic = %d \n" %(ROW_NUM,COL_NUM,DENSITY,WEIGHT,ZERO_HEURISTIC))
    ofile.write("path cost is %d \n"%(pc))
    ofile.write("total number of expansions is %d \n"%(numExpansions))
    ofile.write("total number of steps is %d \n"%(numSteps))
    ofile.write("path : (start) -> ")
    for node in path:
        ofile.write("(%d,%d) ->  "%(node[0],node[1]))
    ofile.write("(end)\n\n")
    
    
    
       
def main():
    global environmen_file_path, MAP
    parser = argparse.ArgumentParser(description='CS 440 A_star search')
    parser.add_argument('-f','--filename', help='File for map', required=True)
    parser.add_argument('-x1','--startx', help='Start position x', required=True)
    parser.add_argument('-y1','--starty', help='Start position y', required=True)
    parser.add_argument('-x2','--goalx', help='goal position x', required=True)
    parser.add_argument('-y2','--goaly', help='goal position y', required=True)
   


    # get the arguments into global variables
    args = vars(parser.parse_args())
    environment_file_path = args['filename']
    startX = int(args['startx'])
    startY = int(args['starty'])
    goalX = int(args['goalx'])
    goalY = int(args['goaly'])
    
    processMap(environment_file_path)
    pc, p = PathFinder(MAP,startX,startY,goalX,goalY)
    output_to_file(p,pc)

if __name__ == "__main__":
    main()

