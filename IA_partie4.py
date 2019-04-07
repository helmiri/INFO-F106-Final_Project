import numpy as np
import random
from config import Parameter, parameter


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def swish(x):
    return x * sigmoid(x)


def swish_deriv(x):
    """
    Derivative of swish
    """
    return swish(x) + sigmoid(x) * (1 - swish(x))


def activation(x, derivative):
    """
    Function selected via a DropDown menu in GUI.
        - args:
            - x: Number/numpy array to apply the function on
            - derivative: Boolean set to True if we want to apply the derivative

        - return:
            - res: Result after derivative is applied

    DISCLAIMER: SWISH and ReLU don't work:
        - With SWISH: Training is ok at the beginning but then an OverFlow error occurs setting the probabilities to NAN
        - With ReLU : Same as SWISH but the probabilities are set to 0
    """

    if parameter.activation == 'Sigmoid':
        if derivative:
            res = x * (1 - sigmoid(x))
        else:
            res = sigmoid(x)
    elif parameter.activation == 'ReLU':
        if derivative:
            if isinstance(x, np.float64):
                res = 1 if x > 0 else 0
            else:
                res = np.copy(x)
                res[res < 0] = 0
                res[res > 0] = 1
        else:
            res = np.maximum(0, x)
    else:
        if derivative:
            res = swish_deriv(x)
        else:
            res = swish(x)
    return res


def initWeights(nb_rows, nb_columns):
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:, 0]
    return (W_int, W_out)


def forwardPass(s, NN):
    W_int = NN[0]
    W_out = NN[1]
    P_int = activation(np.dot(W_int, s), False)
    p_out = activation(P_int.dot(W_out), False)
    return p_out


def backpropagation(s, NN, delta, learning_strategy=None):
    if learning_strategy is None:
        return None

    W_int = NN[0]
    W_out = NN[1]
    P_int = activation(np.dot(W_int, s), False)
    p_out = activation(P_int.dot(W_out), False)
    grad_out = activation(p_out, True)
    grad_int = activation(P_int, True)
    Delta_int = grad_out * W_out * grad_int
    if learning_strategy[0] == 'Q-Learning':
        alpha = learning_strategy[1]
        W_int -= alpha * delta * np.outer(Delta_int, s)
        W_out -= alpha * delta * grad_out * P_int
    elif learning_strategy[0] == 'TD-Lambda' or 'Q-Lambda' or 'DQ-Lambda':
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

    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-Learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-Lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-Lambda')
    DQ_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'DQ-Lambda')


    p_out_s = forwardPass(s, NN)
    greedy = random.random() > parameter.epsilon

    # Find the best move to later compare it with the non-greedy one Q_lambda/DQ_lambda chose
    # to decide whether or not apply the strategy's gimmick
    if greedy or Q_learning or Q_lambda or DQ_lambda:
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
            if best_value is None or c * val > c * best_value:
                best_moves = [m]
                best_value = val if parameter.sigma == 0 else x
            elif val == best_value:
                best_moves.append(m)
    
    if greedy or Q_lambda or DQ_lambda:
        new_s = best_moves[random.randint(0, len(best_moves) - 1)]
        best_s = new_s 
        if DQ_lambda and greedy:
            # Increase lambda
            # V1 of the strategy used a function to gradually increase the lambda
            # It is based on the derivative of sigmoid slightly modified to make the max be 0.9 with a steeper curvature
            # parameter.lamb = 3.6 * sigmoid((3 * parameter.lamb) - 0.9) * (1 - sigmoid((3 * parameter.lamb) - 0.9))
            if p_out_s < best_value:
                parameter.lamb = 0.9

    if not greedy:
        new_s = moves[random.randint(0, len(moves) - 1)]
        if parameter.epsilon_adaptive and learning_strategy is not None:
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

    if learning_strategy is not None:
        if Q_learning:
            delta = p_out_s - best_value
        elif TD_lambda or Q_lambda or DQ_lambda:
            if greedy:
                p_out_new_s = best_value
            else:
                p_out_new_s = forwardPass(new_s, NN)
            delta = p_out_s - p_out_new_s

            if Q_lambda or DQ_lambda:
                chosen_greedy = np.array_equal(new_s, best_s)
                if not chosen_greedy and Q_lambda:
                    # Fill Z_int, Z_out with 0
                    learning_strategy[3].fill(0)
                    learning_strategy[4].fill(0)
                elif not chosen_greedy and DQ_lambda:
                    # Decrease lambda
                    parameter.lamb = sigmoid(2 * parameter.lamb) - 0.5
        backpropagation(s, NN, delta, learning_strategy)

    return new_s


def endGame(s, won, NN, learning_strategy):

    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-Lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-Lambda')
    DQ_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'DQ-Lambda')
    # on met à jour les poids si nécessaire
    if learning_strategy is not None:
        p_out_s = forwardPass(s, NN)
        delta = p_out_s - won
        backpropagation(s, NN, delta, learning_strategy)
        if learning_strategy[0] != 'Q-Learning':
            learning_strategy[3].fill(0)
            learning_strategy[4].fill(0)
