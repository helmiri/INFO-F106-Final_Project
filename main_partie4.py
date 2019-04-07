
'''
    File name: main_partie4.py
    Date: 07/04/2019

    Modifications:
        - Replaced global variables with Parameter class in config.py
        - Deleted clearScreen() and main() as they were unnecessary
        - Rewritten waitForKey() to wait for keyPressEvents coming from Board class in config.py
        - Rewritten display() to update the board in config.py
        - Modified progressBar() to update QProgressBar() in config.py
'''
import copy, os, time, sys, random, time
from select import select
import numpy as np
from IA_partie4 import *
from config import super_awesome_board, parameter, loading_bar
from PyQt5 import QtCore



# KEYS
UP = QtCore.Qt.Key_Up
DOWN = QtCore.Qt.Key_Down
LEFT = QtCore.Qt.Key_Left
RIGHT = QtCore.Qt.Key_Right
# NB: diagonal jumps are usually done using arrow keys by going to the opponent's position first, below: alternative keys
UP_LEFT = QtCore.Qt.Key_D
UP_RIGHT = QtCore.Qt.Key_F
DOWN_LEFT = QtCore.Qt.Key_C
DOWN_RIGHT = QtCore.Qt.Key_V
QUIT = QtCore.Qt.Key_Q
WALL = QtCore.Qt.Key_W
CANCEL = QtCore.Qt.Key_A
PLACE_DOWN = QtCore.Qt.Key_Return


def waitForKey():
    # Enter a loop until a key is pressed

    parameter.running = True
    parameter.closed = False
    while parameter.running and not parameter.closed:
        QtCore.QCoreApplication.processEvents()
        if parameter.running:
            time.sleep(0.1)
        elif parameter.closed:
            key = QUIT
        else:
            key = parameter.keyPressed
    return key


def wait(timeout):
    rlist, wlist, xlist = select([sys.stdin], [], [], timeout)

def progressBar(i, n):
    if int(100 * i / n) > int(100 * (i - 1) / n):
        loading_bar.setValue(int(100 * i / n))

class Player_Human():
    def __init__(self, name='Humain'):
        self.name = name
        self.color = None # white (0) or black(1)
        self.score = 0

    def makeMove(self, board):
        moves = listMoves(board, self.color)
        lmoves = [ listEncoding(m) for m in moves ]
        lboard = listEncoding(board)
        msg = 'Choose a move: \n\nArrows: Move\nW: Place a wall\nQ: Quit'
        
        D = [ [LEFT, [-1,0], None], [RIGHT, [1,0], None], [UP, [0,1], None], [DOWN, [0,-1], None ], # one step moves
              [LEFT, [-2,0], None], [RIGHT, [2,0], None], [UP, [0,2], None], [DOWN, [0,-2], None ], # jumps
              [UP_LEFT, [-1,1], None], [UP_RIGHT, [1,1], None], [DOWN_LEFT, [-1,-1], None], [DOWN_RIGHT, [1,-1], None] ] # diagonal moves
        for i in range(len(D)):
            for j in range(len(lmoves)):
                m = moves[j]
                lm = lmoves[j]
                if list(np.array(lm[self.color])) == list(np.array(lboard[self.color]) + np.array(D[i][1])):
                    D[i][2] = m
                    break
        wall_moves = [ lm for lm in lmoves if lm[self.color]==lboard[self.color] ]
        wall_coord = [ [], [] ]
        wall_hv = [ [], [] ]
        for lm in wall_moves:
            i = int(lm[2] == lboard[2])
            wall_hv[i].append(lm)
            for c in lm[2+i]:
                if c not in lboard[2+i]:
                    break
            wall_coord[i].append(c)

        quit = False
        while not quit:
            display(board, msg)
            key = waitForKey()
            if key == QUIT:
                quit = True
                break
            # player changes position:
            for i in range(len(D)):
                if key==D[i][0]:
                    if not (D[i][2] is None):
                        return D[i][2]
                    elif (i <= 3) and (D[i+4][2] is None):
                        p = np.array(lboard[self.color]) + np.array(D[i][1])
                        q = np.array(lboard[(self.color+1)%2])
                        if p.tolist() == q.tolist():
                            # we moved to the opponent's position but the jump is blocked, we check if some diagonal move is possible
                            s = np.array(D[i][1])
                            diagonal_jump_feasible = False
                            for j in range(8, 12):
                                if not (D[j][2] is None):
                                    r = np.array(D[j][1])
                                    if r[0]==s[0] or r[1]==s[1]:
                                        diagonal_jump_feasible = True
                            if diagonal_jump_feasible:
                                halfway = board.copy()
                                halfway[self.color*parameter.size**2:self.color*parameter.size**2 + parameter.size**2] = halfway[((self.color+1)%2)*parameter.size**2:((self.color+1)%2)*parameter.size**2 + parameter.size**2]
                                display(halfway, 'Diagonal jump\n\nChoose a destination')
                                second_key = waitForKey()
                                diagonal_jump = False
                                for j in range(4):
                                    if second_key==D[j][0]:
                                        t = np.array(D[j][1])
                                        r = s + t
                                        if abs(r[0])==1 and abs(r[1])==1:
                                            # diagonal jump selected, we check if that jump is feasible
                                            for k in range(8, 12):
                                                if r.tolist() == D[k][1] and not (D[k][2] is None):
                                                    key = D[k][0]
                                                    diagonal_jump = True
                                                    break
                                if not diagonal_jump:
                                    display(board, msg)
                                    

            # player puts down a wall
            if key == WALL and lboard[4 + self.color] > 0 and len(wall_moves) > 0:
                msg = "Wall placement\n\nArrows: Move\nW: Change orientation\nENTER: Place down\nA: Cancel wall placement\nQ: Quit"
                j = 0
                if len(wall_hv[0])>0:
                    h = 0
                else:
                    h = 1
                while not quit:
                    i = lmoves.index(wall_hv[h][j])
                    display(moves[i], msg)
                    key = waitForKey()
                    if key==QUIT:
                        quit = True
                        break 
                    if key == CANCEL:
                        display(board, msg)
                        break
                    elif key == WALL:
                        if len(wall_hv[(h+1)%2])>0:
                            c = wall_coord[h][j]
                            h=(h+1)%2
                            if c in wall_coord[h]:
                                j = wall_coord[h].index(c)
                            else:
                                best_d = wall_coord[h][0]
                                min_val = (abs(best_d[0]-c[0]) + abs(best_d[1]-c[1]))**2
                                for d in wall_coord[h]:
                                    val = (abs(d[0]-c[0]) + abs(d[1]-c[1]))**2
                                    if val < min_val:
                                        min_val = val
                                        best_d = d
                                j = wall_coord[h].index(best_d)
                    elif key == LEFT:
                        j = (j - 1) % len(wall_hv[h])
                    elif key == RIGHT:
                        j = (j + 1) % len(wall_hv[h])
                    elif key == UP:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, len(wall_coord[h])):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] > c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == DOWN:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, -1, -1):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] < c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == PLACE_DOWN:
                        return moves[i]
        if quit:
            return None


    def endGame(self, board, won):
        # Reset Board. Simplest way I could find
        parameter.endGame = True
        super_awesome_board.display([[0, 0], [0, 0], [], [], parameter.walls, parameter.walls])
        pass



class Player_AI():
    def __init__(self, NN, eps, learning_strategy, name='IA'):
        self.name = name
        self.color = None # white (0) or black(1)
        self.score = 0
        self.NN = NN
        self.eps = eps
        self.learning_strategy = learning_strategy

    def makeMove(self, board):
        return makeMove(listMoves(board, self.color), board, self.color, self.NN, self.eps, self.learning_strategy)

    def endGame(self, board, won):
        if not parameter.training:
            parameter.endGame = True
            super_awesome_board.display([[0, 0], [0, 0], [], [], parameter.walls, parameter.walls])
        endGame(board, won, self.NN, self.learning_strategy)


def listEncoding(board):
    # outputs list encoding of board:
    # [ [i, j], [k, l], list_of_horizontal_walls, list_of_vertical_walls, walls_left_p1, walls_left_p2 ]
    # where [i, j] position of white player and [k, l] position of black player
    # and each wall in lists of walls is of the form [a, b] where [a,b] is the south-west square
    pos = [None,None]
    coord = [ [None,None], [None,None]]
    walls = [ [], [] ]
    walls_left = [ None, None ]
    for i in range(2):
        pos[i] = board[i*parameter.size**2:(i+1)*parameter.size**2].argmax()
        coord[i][0] = pos[i]%parameter.size
        coord[i][1] = pos[i]//parameter.size
        for j in range((parameter.size-1)**2):
            if board[2*parameter.size**2 + i*(parameter.size-1)**2 + j]==1:
                walls[i].append( [j%(parameter.size-1), j//(parameter.size-1)] )
        walls_left[i] = board[2 * parameter.size ** 2 + 2 * (parameter.size - 1)** 2 + i * (parameter.walls + 1):2 * parameter.size ** 2 + 2 * (parameter.size - 1)** 2 + (i + 1) * (parameter.walls + 1)].argmax()
    return [ coord[0], coord[1], walls[0], walls[1], walls_left[0], walls_left[1] ]


def display(board, msg=None):
    lboard = listEncoding(board)
    super_awesome_board.message.setPlainText(msg)
    super_awesome_board.display(lboard)
    QtCore.QCoreApplication.processEvents()


def eachPlayerHasPath(board):
    # heuristic when at most one wall
    nb_walls = board[2*parameter.size**2:2*parameter.size**2 + 2*(parameter.size-1)**2].sum()
    if nb_walls <= 2:
        # there is always a path when there is at most one wall
        return True
    # checks whether the two players can each go to the opposite side
    pos = [None, None]
    coord = [ [None,None], [None,None]]
    for i in range(2):
        pos[i] = board[i*parameter.size**2:(i+1)*parameter.size**2].argmax()
        coord[i][0] = pos[i]%parameter.size
        coord[i][1] = pos[i]//parameter.size
        coord[i] = np.array(coord[i])
    steps = [ (1,0), (0,1), (-1,0), (0,-1) ]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    for i in range(2):
        A = np.zeros((parameter.size,parameter.size), dtype='bool')   # TO DO: this could be optimized
        S = [ coord[i] ]  # set of nodes left to treat
        finished = False
        while len(S)>0 and not finished:
            c=S.pop()
            # NB: In A we swap rows and columns for simplicity
            A[c[1]][c[0]]=True
            for k in range(4):
                if parameter.g[c[0]][c[1]][k]==1:
                    s = steps[k]
                    new_c = c + s
                    # test whether we reached the opposite row
                    if i == 0:
                        if new_c[1]==parameter.size-1:
                            finished = True
                            break
                    else:
                        if new_c[1]==0:
                            return True
                    # otherwise we continue exploring
                    if A[new_c[1]][new_c[0]] == False:
                        # heuristic, we give priority to moves going up (down) in case player is white (black)
                        if i == 0:
                            if k == 1:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
                        else:
                            if k == 3:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
        if not finished:
            return False
    return True



def canMove(board, coord, step):
    # returns True if there is no wall in direction step from pos, and we stay in the board
    # NB: it does not check whether the destination is occupied by a player
    new_coord = coord + step
    in_board = new_coord.min() >= 0 and new_coord.max() <= parameter.size-1
    if not in_board:
        return False
    if parameter.walls > 0:
        if step[0] == -1:
            L = []
            if new_coord[1] < parameter.size-1:
                L.append(2*parameter.size**2 + (parameter.size-1)**2 + new_coord[1]*(parameter.size-1) + new_coord[0])
            if new_coord[1] > 0:
                L.append(2*parameter.size**2 + (parameter.size-1)**2 + (new_coord[1]-1)*(parameter.size-1) + new_coord[0])
        elif step[0] == 1:
            L = []
            if coord[1] < parameter.size-1:
                L.append(2*parameter.size**2 + (parameter.size-1)**2 + coord[1]*(parameter.size-1) + coord[0])
            if coord[1] > 0:
                L.append(2*parameter.size**2 + (parameter.size-1)**2 + (coord[1]-1)*(parameter.size-1) + coord[0])
        elif step[1] == -1:
            L = []
            if new_coord[0] < parameter.size-1:
                L.append(2*parameter.size**2 + new_coord[1]*(parameter.size-1) + new_coord[0])
            if new_coord[0] > 0:
                L.append(2*parameter.size**2 + new_coord[1]*(parameter.size-1) + new_coord[0]-1)
        elif step[1] == 1:
            L = []
            if coord[0] < parameter.size-1:
                L.append(2*parameter.size**2 + coord[1]*(parameter.size-1) + coord[0])
            if coord[0] > 0:
                L.append(2*parameter.size**2 + coord[1]*(parameter.size-1) + coord[0]-1)
        else:
            print('step vector', step, 'is not valid')
            quit(1)
        if sum([ board[j] for j in L ]) > 0:
            # move blocked by a wall
            return False
    return True

def computeGraph(board=None):
    # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
    pos_steps = [ (1,0), (0,1) ]
    for i in range(len(pos_steps)):
        pos_steps[i] = np.array(pos_steps[i])
    g = np.zeros((parameter.size,parameter.size,4))
    for i in range(parameter.size):
        for j in range(parameter.size):
            c = np.array([i,j])
            for k in range(2):
                s = pos_steps[k]
                if board is None:
                    # initial setup
                    new_c = c + s
                    if new_c.min() >= 0 and new_c.max() <= parameter.size-1:
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k+2] = 1
                else:
                    if canMove(board, c, s):
                        new_c = c + s
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k+2] = 1
    return g

def listMoves(board, current_player):
    if current_player not in [0,1]:
        print('error in function listMoves: current_player =', current_player)
    pn = current_player
    steps = [ (-1,0), (1,0), (0,-1), (0,1) ]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    moves = []
    pos = [None, None]
    coord = [None, None]
    for i in range(2):
        pos[i] = board[i*parameter.size**2:(i+1)*parameter.size**2].argmax()
        coord[i] = np.array([ pos[i]%parameter.size, pos[i]//parameter.size ])
        pos[i] += pn*parameter.size**2     # offset for black player
    P = [] # list of new boards (each encoded as list bits to switch)
    # current player moves to another position
    for s in steps:
        if canMove(board, coord[pn], s):
            new_coord = coord[pn] + s
            new_pos = pos[pn] + s[0] + parameter.size*s[1]
            occupied = np.array_equal(new_coord, coord[(pn+1)%2])
            if not occupied:
                P.append([ pos[pn], new_pos ])  # new board is obtained by switching these two bits
            else:
                can_jump_straight = canMove(board, new_coord, s)
                if can_jump_straight:
                    new_pos = new_pos + s[0] + parameter.size*s[1]
                    P.append([ pos[pn], new_pos ])
                else:
                    if s[0]==0:
                        D = [ (-1, 0), (1, 0) ]
                    else:
                        D = [ (0, -1), (0, 1) ]
                    for i in range(len(D)):
                        D[i] = np.array(D[i])
                    for d in D:
                        if canMove(board, new_coord, d):
                            final_pos = new_pos + d[0] + parameter.size*d[1]
                            P.append([ pos[pn], final_pos ])
    # current player puts down a wall
    # TO DO: Speed up this part: it would perhaps be faster to directly discard intersecting walls based on existing ones
    nb_walls_left = board[2*parameter.size**2 + 2*(parameter.size-1)**2 + pn*(parameter.walls+1):2*parameter.size**2 + 2*(parameter.size-1)**2 + (pn+1)*(parameter.walls+1)].argmax()
    ind_walls_left = 2*parameter.size**2 + 2*(parameter.size-1)**2 + pn*(parameter.walls+1) + nb_walls_left
    if nb_walls_left > 0:
        for i in range(2*(parameter.size-1)**2):
            pos = 2*parameter.size**2 + i
            L = [ pos ]  # indices of walls that could intersect
            if i < (parameter.size-1)**2:
                # horizontal wall
                L.append(pos+(parameter.size-1)**2)  # vertical wall on the same 4-square
                if i%(parameter.size-1)>0:
                    L.append(pos-1)
                if i%(parameter.size-1)<parameter.size-2:
                    L.append(pos+1)
            else:
                # vertical wall
                L.append(pos-(parameter.size-1)**2)  # horizontal wall on the same 4-square
                if (i-(parameter.size-1)**2)//(parameter.size-1)>0:
                    L.append(pos-(parameter.size-1))
                if (i-(parameter.size-1)**2)//(parameter.size-1)<parameter.size-2:
                    L.append(pos+(parameter.size-1))
            nb_intersecting_wall = sum([ board[j] for j in L ])
            if nb_intersecting_wall==0:
                board[pos] = 1
                # we remove the corresponding edges from parameter.g
                if i < (parameter.size-1)**2:
                    # horizontal wall
                    a, b = i%(parameter.size-1), i//(parameter.size-1)
                    E = [ [a,b,1], [a,b+1,3], [a+1,b,1], [a+1,b+1,3] ]
                else:
                    # vertical wall
                    a, b = (i - (parameter.size-1)**2)%(parameter.size-1), (i - (parameter.size-1)**2)//(parameter.size-1)
                    E = [ [a,b,0], [a+1,b,2], [a,b+1,0], [a+1,b+1,2] ]
                for e in E:
                    parameter.g[e[0]][e[1]][e[2]] = 0
                if eachPlayerHasPath(board):
                    P.append([pos, ind_walls_left-1, ind_walls_left])  # put down the wall and adapt player's counter
                board[pos] = 0
                # we add back the two edges in parameter.g
                for e in E:
                    parameter.g[e[0]][e[1]][e[2]] = 1
    # we create the new boards from P
    for L in P:
        new_board = board.copy()
        for i in L:
            new_board[i] = not new_board[i]
        moves.append(new_board)

    return moves

def endOfGame(board):
    return board[(parameter.size-1)*parameter.size:parameter.size**2].max() == 1 or board[parameter.size**2:parameter.size**2+parameter.size].max() == 1

def startingBoard():
    board = np.array([0]*(2*parameter.size**2 + 2*(parameter.size-1)**2 + 2*(parameter.walls+1)))
    # player positions
    board[ (parameter.size-1)//2 ] = True
    board[ parameter.size**2 + parameter.size*(parameter.size-1) + (parameter.size-1)//2 ] = True
    # wall counts
    for i in range(2):
        board[ 2*parameter.size**2 + 2*(parameter.size-1)**2 + i*(parameter.walls+1) + parameter.walls ] = 1
    return board

def playGame(player1, player2, show = False, delay = 0.0):
    # initialization
    players = [ player1, player2 ]
    board = startingBoard()
    parameter.g = parameter.g_init.copy()
    for i in range(2):
        players[i].color = i
    # main loop
    finished = False
    current_player = 0
    count = 0
    quit = False
    while not finished:
        QtCore.QCoreApplication.processEvents()
        if show:
            msg = ''
            for i in range(2):
                if players[i].name=='IA':
                    # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                    p = forwardPass(board, players[i].NN)
                    super_awesome_board.probablity.setPlainText("AI's estimation : {0:.4f}".format(p))
        new_board = players[current_player].makeMove(board)
        super_awesome_board.current_player.setPlainText('*Black\n  White' if players[current_player].color == 0 else '  Black\n*White')

        # we compute changes of parameter.g (if any) to avoid recomputing parameter.g at beginning of listMoves
        # we remove the corresponding edges from parameter.g
        if not new_board is None:
            v = new_board[2*parameter.size**2:2*parameter.size**2 + 2*(parameter.size-1)**2] - board[2*parameter.size**2:2*parameter.size**2 + 2*(parameter.size-1)**2]
            i = v.argmax()
            if v[i] == 1:
                # a wall has been added, we remove the two corresponding edges of parameter.g
                if i < (parameter.size-1)**2:
                    # horizontal wall
                    a, b = i%(parameter.size-1), i//(parameter.size-1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    # vertical wall
                    a, b = (i - (parameter.size-1)**2)%(parameter.size-1), (i - (parameter.size-1)**2)//(parameter.size-1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    parameter.g[e[0]][e[1]][e[2]] = 0
        board = new_board
        if board is None:
            # human player quit
            quit = True
            finished = True
        elif endOfGame(board):
            players[current_player].score += 1
            white_won = current_player == 0
            players[current_player].endGame(board, white_won)
            if show:
                display(board, msg)
            finished = True
        else:
            current_player = (current_player+1)%2
    return quit

def train(NN, n_train=10000):
    if parameter.learning_strategy == 'Q-Learning':
        learning_strategy1 = (parameter.learning_strategy, parameter.alpha)
        learning_strategy2 = (parameter.learning_strategy, parameter.alpha)
    elif parameter.learning_strategy == 'TD-Lambda' or 'Q-Lambda' or 'DQ-Lambda':
        learning_strategy1 = [parameter.learning_strategy, parameter.alpha, parameter.lamb, np.zeros(NN[0].shape), np.zeros(NN[1].shape)]
        learning_strategy2 = [parameter.learning_strategy, parameter.alpha, parameter.lamb, np.zeros(NN[0].shape), np.zeros(NN[1].shape)]
    base_epsilon = parameter.epsilon
    base_lambda = parameter.lamb
    agent1 = Player_AI(NN, base_epsilon, learning_strategy1, 'agent 1')
    agent2 = Player_AI(NN, base_epsilon, learning_strategy2, 'agent 2')
    # training session
    for j in range(n_train):
        progressBar(j, n_train)
        QtCore.QCoreApplication.processEvents()
        playGame(agent1, agent2)
        # Restore their original values at the start of each game
        # in case they have been modified
        parameter.epsilon = base_epsilon
        parameter.lamb = base_lambda
        parameter.k = 0
        
        

def compare(NN1, filename, n_compare=1000, eps=0.05):
    agent1 = Player_AI(NN1, eps, None, 'agent 1')
    data = np.load(filename)
    NN2 = (data['W1'], data['W2'])
    agent2 = Player_AI(NN2, eps, None, 'agent 2')
    players = [agent1, agent2]
    i = 0
    for j in range(n_compare):
        progressBar(j, n_compare)
        QtCore.QCoreApplication.processEvents()
        playGame(players[i], players[(i+1)%2])
        i = (i+1)%2
    perf = agent1.score / n_compare
    
    return "{0:.2f}".format(perf*100)+'%'


def play(player1, player2, delay=0.2):
    i = 0
    players = [player1, player2]
    quit = False
    while not quit:
        quit = playGame(players[i], players[(i+1)%2], True, delay)
        i = (i + 1) % 2
    parameter.endGame = True
