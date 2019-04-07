
'''
    File name: tournoi.py
    Author   : EL MIRI Hamza
               Based on the work of Gwenaël Joret and Arnaud Pollaris
    Date     : 07/04/2019

    AI config:
        - Activation function: Sigmoid
        - Learning strategy  : DQ-Lambda
        - Espilon Noise      : Yes, sigma = 0.01
        - Epsilon            : Adaptive (0.5)
        - Neurons            : 40
        - Lambda             : 0.9
        - Alpha              : 0.3
        
'''
import copy, os, time, sys, random, time
from select import select
import numpy as np


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
        if show:
            msg = ''
            for i in range(2):
                if players[i].name=='IA':
                    # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                    p = forwardPass(board, players[i].NN)
        new_board = players[current_player].makeMove(board)

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
    learning_strategy1 = [parameter.learning_strategy, parameter.alpha, parameter.lamb, np.zeros(NN[0].shape), np.zeros(NN[1].shape)]
    learning_strategy2 = [parameter.learning_strategy, parameter.alpha, parameter.lamb, np.zeros(NN[0].shape), np.zeros(NN[1].shape)]
    base_epsilon = parameter.epsilon
    base_lambda = parameter.lamb
    agent1 = Player_AI(NN, base_epsilon, learning_strategy1, 'agent 1')
    agent2 = Player_AI(NN, base_epsilon, learning_strategy2, 'agent 2')
    # training session
    for j in range(n_train):
        playGame(agent1, agent2)
        # Restore their original values at the start of each game
        # in case they have been modified
        parameter.epsilon = base_epsilon
        parameter.lamb = base_lambda
        parameter.k = 0


def play(player1, player2, delay=0.2):
    i = 0
    players = [player1, player2]
    quit = False
    while not quit:
        quit = playGame(players[i], players[(i+1)%2], True, delay)
        i = (i + 1) % 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initWeights(nb_rows, nb_columns):
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:, 0]
    return (W_int, W_out)


def forwardPass(s, NN):
    W_int = NN[0]
    W_out = NN[1]
    P_int = sigmoid(np.dot(W_int, s))
    p_out = sigmoid(P_int.dot(W_out))
    return p_out


def backpropagation(s, NN, delta, learning_strategy=None):

    W_int = NN[0]
    W_out = NN[1]
    P_int = sigmoid(np.dot(W_int, s))
    p_out = sigmoid(P_int.dot(W_out))
    grad_out = p_out * (1 - sigmoid(p_out))
    grad_int = P_int * (1 - sigmoid(P_int))
    Delta_int = grad_out * W_out * grad_int
    alpha = learning_strategy[1]
    lamb = learning_strategy[2]

    Z_int = learning_strategy[3]
    Z_out = learning_strategy[4]
    Z_int *= lamb
    Z_int += np.outer(Delta_int, s)

    Z_out *= lamb
    Z_out += grad_out*P_int
    W_int -= alpha*delta*Z_int
    W_out -= alpha*delta*Z_out


def makeMove(moves, s, color, NN, eps, learning_strategy=None):

    p_out_s = forwardPass(s, NN)
    greedy = random.random() > parameter.epsilon

    # Find the best move to later compare it with the non-greedy one Q_lambda/DQ_lambda chose
    # to decide whether or not apply the strategy's gimmick
    best_moves = []
    best_value = None
    c = 1
    if color == 1:
        c = -1
    for m in moves:
        val = forwardPass(m, NN)
        x = val   # Remember the original value

        if parameter.sigma != 0 and learning_strategy is not None:
            # If epsilon with noise is applied, apply the noise on val
            # before choosing a move. The original value of val before the noise is remembered
            # for later use otherwise it will mess up the neural network
            val += np.random.normal(0, parameter.sigma)

        if best_value is None or c * val > c * compare_value:
            best_moves = [m]
            compare_value = val
            best_value = x
        elif val == best_value:
            best_moves.append(m)
    
    # Greedy move to use later for comparison
    new_s = best_moves[random.randint(0, len(best_moves) - 1)]
    best_s = new_s
    
    if greedy:
        # Increase lambda
        # V1 of the strategy used a function to gradually increase the lambda
        # It is based on the derivative of sigmoid slightly modified to make the max be 0.9 with a steeper curvature
        # parameter.lamb = 3.6 * sigmoid((3 * parameter.lamb) - 0.9) * (1 - sigmoid((3 * parameter.lamb) - 0.9))
        if p_out_s < best_value:
            parameter.lamb = 0.9

    if not greedy:
        new_s = moves[random.randint(0, len(moves) - 1)]
        parameter.k += 1
        if parameter.k == parameter.exploration_limit:
            # after exploration_limit number of times a non-greedy move has been chosen,
            # adjust epsilon
            max_current = forwardPass(new_s, NN)
            diff = max_current - p_out_s
            if diff > 0:
                # Decrease Epsilon given that our chances of winning have increased
                parameter.epsilon = sigmoid(2 * parameter.epsilon) - 0.5
            elif diff < 0:
                # Restore base epsilon
                parameter.epsilon = eps
            parameter.k = 0

    if greedy:
        p_out_new_s = best_value
    else:
        p_out_new_s = forwardPass(new_s, NN)
    delta = p_out_s - p_out_new_s

    chosen_greedy = np.array_equal(new_s, best_s)
    if not chosen_greedy:
        # Decrease lambda
        parameter.lamb = sigmoid(2 * parameter.lamb) - 0.5
    backpropagation(s, NN, delta, learning_strategy)

    return new_s


def endGame(s, won, NN, learning_strategy):

    DQ_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'DQ-Lambda')
    p_out_s = forwardPass(s, NN)
    delta = p_out_s - won
    backpropagation(s, NN, delta, learning_strategy)
    learning_strategy[3].fill(0)
    learning_strategy[4].fill(0)


class Parameter:
    """
    Class containing all of the parameters that will be used
    """

    def __init__(self):
        # AI
        self.size = 5
        self.walls = 3
        self.epsilon = 0.5
        self.alpha = 0.3
        self.lamb = 0.9
        self.learning_strategy = 'DQ-Lambda'
        self.g = None
        self.g_init = None
        self.neurons = 40
        self.sigma = 0.01
        self.NN = None

        # Part4 related
        self.max_prev = 0
        self.k = 0
        self.exploration_limit = 5
        self.epsilon_adaptive = True
        self.max_value = 0


if __name__ == '__main__':
    print('SIGMOID')
    arguments = sys.argv
    filename = arguments[-1]
    training = int(arguments[-2])
    parameter = Parameter()

    parameter.g_init = computeGraph()
    parameter.g = parameter.g_init.copy()
    parameter.NN = createNN(2 * parameter.size ** 2 + 2 * (parameter.size - 1) ** 2 + 2 * (parameter.walls + 1), parameter.neurons)

    train(parameter.NN, training)
    np.savez(filename, N=5, WALLS=3, W1=parameter.NN[0], W2=parameter.NN[1])
