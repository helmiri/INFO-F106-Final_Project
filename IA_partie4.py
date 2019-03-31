import numpy as np
import random
from config import Parameter, parameter

def sigmoid(x):
    """Cette fonction calcule la valeur de la fonction sigmoide (de 0 à 1) pour un nombre réel donné.
    Il est à noter que x peut également être un vecteur de type numpy array, dans ce cas, la valeur de la sigmoide correspondante à chaque réel du vecteur est calculée.
    En retour de la fonction, il peut donc y avoir un objet de type numpy.float64 ou un numpy array."""
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def swish(x):
    return x * sigmoid(x)

def swish_deriv(x):
    return x * sigmoid(x) * (1 - sigmoid(x)) + sigmoid(x)

def activation(x, derivative):
    if parameter.activation == 'Sigmoid':
        if derivative:
            res = sigmoid(x) * (1 - sigmoid(x))
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
    """Fonction destinée à initialiser les poids d'une matrice de genre (nb_rows * nb_columns) pour un réseau de neurones sur base d'une distribution normale de moyenne 0 et d'un écart-type de 0.0001."""
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    """Fonction permettant de créer un réseau de neurones en fonction de la taille de la couche d'entrée (n_input) et de la couche intermédiaire (n_hidden)
    Le réseau de neurones créé est ensuite retourné sous la forme d'un tuple de 2 numpy array contenant respectivement
    les coefficients de poids liant la couche d'entrée à la couche intermédiaire et les coefficients de poids liant la couche intermédiaire à la couche de sortie.
    """
    W_int = initWeights(n_hidden, n_input)
    # W2 est traité comme vecteur est non comme une matrice Hx1 (simplifie certaines ecritures)
    W_out = initWeights(n_hidden, 1)[:, 0]
    return (W_int, W_out)


def forwardPass(s, NN):
    """Cette fonction permet d'utiliser un réseau de neurones NN pour estimer la probabilité de victoire finale du joueur blanc pour un état (s) donné du jeu."""
    W_int = NN[0]
    W_out = NN[1]
    # P_int = sigmoid(np.dot(W_int, s))
    # p_out = sigmoid(P_int.dot(W_out))
    P_int = activation(np.dot(W_int, s), False)
    p_out = activation(P_int.dot(W_out), False)
    if parameter.sigma != 0:
        p_out += np.random.normal(0, parameter.sigma)
    return p_out 


def backpropagation(s, NN, delta, learning_strategy=None):
    """Fonction destinée à réaliser la mise à jour des poids d'un réseau de neurones (NN). Cette mise à jour se fait conformément à une stratégie d'apprentissage (learning_strategy)
    pour un état donné (s) du jeu.
    Le delta est la différence de probabilité de gain estimée entre deux états successif potentiels du jeu.
    La stratégie d'apprentissage peut soit être None, soit il s'agit d'un tuple de la forme ('Q-learning', alpha) où alpha est le learning_rate (une valeur entre 0 et 1 inclus),
    soit il s'agit d'un tuple de la forme ('TD-lambda', alpha, lamb, Z_int, Z_out) où alpha est le learning_rate, lamb est la valeur de lambda (entre 0 et 1 inclus) et
    Z_int et Z_out contiennent les valeurs de l'éligibility trace associées respectivement aux différents poids du réseau de neurones.
    La fonction de backpropagation ne retourne rien de particulier (None) mais les poids du réseau de neurone NN (W_int, W_out) peuvent être modifiés,
    idem pour l'eligibility trace (Z_int et Z_out) dans le cas où la stratégie TD-lambda est utilisée.
    """
    # remarque : les operateurs +=, -=, et *= changent directement un numpy array sans en faire une copie au prealable, ceci est nécessaire
    # lorsqu'on modifie W_int, W_out, Z_int, Z_out ci-dessous (sinon les changements seraient perdus après l'appel)
    if learning_strategy is None:
        # pas de mise à jour des poids
        return None
    W_int = NN[0]
    W_out = NN[1]
    P_int = activation(np.dot(W_int, s), False)
    p_out = activation(P_int.dot(W_out), False)
    grad_out = activation(p_out, True)
    grad_int = activation(P_int, True)

    Delta_int = grad_out*W_out*grad_int
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha*delta*np.outer(Delta_int, s)
        W_out -= alpha*delta*grad_out*P_int
    elif learning_strategy[0] == 'TD-lambda':
        alpha = learning_strategy[1]
        lamb = learning_strategy[2]
        Z_int = learning_strategy[3]
        Z_out = learning_strategy[4]
        Z_int *= lamb
        Z_int += np.outer(Delta_int, s)
        # remarque : si au lieu des deux operations ci-dessus nous avions fait
        #    Z_int = lamb*Z_int + np.outer(Delta_int,s)
        # alors cela aura créé un nouveau numpy array, en particulier learning_strategy[3] n'aurait pas été modifié
        Z_out *= lamb
        Z_out += grad_out*P_int
        W_int -= alpha*delta*Z_int
        W_out -= alpha*delta*Z_out


def makeMove(moves, s, color, NN, eps, learning_strategy=None):
    """Fonction appelée pour que l'IA choisisse une action (le nouvel état new_s est retourné)
     à partir d'une liste d'actions possibles "moves" (c'est-à-dire une liste possible d'états accessibles) à partir d'un état actuel (s)
     Un réseau de neurones (NN) est nécessaire pour faire ce choix quand le mouvement n'est pas du hasard.
     Dans le cas greedy (non aléatoire), la couleur du joueur dont c'est le tour sera utilisée pour déterminer s'il faut retenir le meilleur ou le pire coup du point de vue du joueur blanc.
     Le cas greedy survient avec une probabilité 1-eps.
     La stratégie d'apprentissage fait référence à la stratégie utilisée dans la fonction de backpropagation (cfr la fonction de backpropagation pour une description)
     """
    Q_learning = (not learning_strategy is None) and (
        learning_strategy[0] == 'Q-Learning')
    TD_lambda = (not learning_strategy is None) and (
        learning_strategy[0] == 'TD-Lambda')
    Q_lambda = (not learning_strategy is None) and (
        learning_strategy[0] == 'Q-Lambda')
    # Epsilon greedy
    greedy = random.random() > eps
    # dans le cas greedy, on recherche le meilleur mouvement (état) possible. Dans le cas du Q-learning (même sans greedy), on a besoin de connaître
    # la probabilité estimée associée au meilleur mouvement (état) possible en vue de réaliser la backpropagation.
    if greedy or Q_learning or Q_lambda:
        best_moves = []
        best_value = None
        c = 1
        if color == 1:
            # au cas où c'est noir qui joue, on s'interessera aux pires coups du point de vue de blanc
            c = -1
        for m in moves:
            val = forwardPass(m, NN)
            # si noir joue, c'est comme si on regarde alors si val < best_value
            if best_value == None or c*val > c*best_value:
                best_moves = [m]
                best_value = val
            elif val == best_value:
                best_moves.append(m)


    if greedy or (Q_lambda and not greedy):
        # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
        new_s = best_moves[random.randint(0, len(best_moves) - 1)]
        best_s = new_s
    if not greedy:
        # on choisit un mouvement au hasard
        new_s = moves[random.randint(0, len(moves) - 1)]
        if Q_lambda:
            if np.array_equal(new_s, best_s):
                learning_strategy[3] *= 0
                learning_strategy[4] *= 0
                print("what")


    # on met à jour les poids si nécessaire
    if learning_strategy is not None:
        p_out_s = forwardPass(s, NN)
        if Q_learning:
            delta = p_out_s - best_value
        elif TD_lambda or Q_lambda:
            if greedy:
                p_out_new_s = best_value
            else:
                p_out_new_s = forwardPass(new_s, NN)
            delta = p_out_s - p_out_new_s

        backpropagation(s, NN, delta, learning_strategy)

    return new_s


def endGame(s, won, NN, learning_strategy):
    """Cette fonction est appelée en fin de partie, pour une dernière mise à jour des poids lorsqu'une stratégie d'apprentissage est appliquée.
    Les arguments s, NN et learning strategy sont les mêmes que ceux de la fonction makeMove() décrite ci-dessus.
    Ce qui change, c'est le booléen "won" indiquant si le joueur blanc a gagné (True) ou perdu (False). On est ici certain de la probabilité de gain de blanc (1 ou 0).
    Comme précédemment, si une stratégie d'apprentissage est appliquée, un delta sera calculé : ce sera ici sur la base de la probabilité associée au dernier état
    et de la conséquence associée (victoire ou défaite de blanc).
    Il servira à la mise à jour des poids du réseau de neurones.
    Dans le cas du TD-lambda, les éligibility traces sont aussi remises à 0 à la fin.
    Donc, cette fonction ne retourne rien (None) mais elle peut avoir un impact sur les valeurs des poids de NN
    et sur l'eligibility trace associée (quand TD-lambda est utilisée)
    """
    Q_learning = (not learning_strategy is None) and (
        learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (
        learning_strategy[0] == 'TD-lambda')
    # on met à jour les poids si nécessaire
    if Q_learning or TD_lambda:
        p_out_s = forwardPass(s, NN)
        delta = p_out_s - won
        backpropagation(s, NN, delta, learning_strategy)
        if TD_lambda:
            # on remet les eligibility traces à 0 en prévision de la partie suivante
            learning_strategy[3].fill(0)  # remet Z_int à 0
            learning_strategy[4].fill(0)  # remet Z_out à 0
