'''
    File name: config.py
    Author: El Miri Hamza
    ID: 000479603
    Date: 07/04/2019

    Function:
        - Contains variables shared across partie3.py and main_partie2.py
        - Creates a UI for the board
'''

from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QProgressBar, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtCore import QRectF, Qt


class Parameter:
    """
    Class containing all of the parameters that will be used
    """

    def __init__(self):
        # AI
        self.size = 5
        self.walls = 1
        self.epsilon = 0.3
        self.alpha = 0.4
        self.lamb = 0.9
        self.learning_strategy = 'Q-Learning'
        self.g = None
        self.g_init = None
        self.neurons = 40
        self.activation = "Sigmoid"
        self.eps_decrease = 0
        self.eps_end_value = 0
        self.sigma = 0

        # Game
        self.train = 10000
        self.compare = 10000
        self.NN = None
        self.board = None

        # Wether or not to update the board
        self.training = True

        # Control thingies
        self.running = True
        self.keyPressed = None
        self.closed = False
        self.endGame = False

        #
        self.max_prev = 0
        self.k = 0
        self.l = 5
        self.epsilon_adaptive = False


class Board(QMainWindow):
    """
    Board window using QGraphicsView and QGraphicsScene
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Make AI Great Again")
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.message = QGraphicsTextItem()
        self.scene.addItem(self.message)

        self.current_player = QGraphicsTextItem()
        self.scene.addItem(self.current_player)

        self.probablity = QGraphicsTextItem()
        self.scene.addItem(self.probablity)

        # Create player
        self.player1 = QGraphicsEllipseItem(QRectF(0, 0, 90, 90))
        self.player2 = QGraphicsEllipseItem(QRectF(0, 0, 90, 90))
        self.player2.setBrush(Qt.black)
        self.player1.setBrush(Qt.white)

    def initBoard(self):
        # Create board cells
        self.item_list = list()
        for i in range(parameter.size):
            for j in range(parameter.size):
                cell = QGraphicsRectItem(QRectF(j * 100, i * 100, 100, 100))
                cell.setBrush(QBrush(Qt.gray))
                self.item_list.append(cell)

        for item in self.item_list:
            self.scene.addItem(item)

        # Create all of the walls that will be used
        self.walls_on_board = {'Horizontal': [], 'Vertical': []}
        self.walls_on_board['Horizontal'] = self.create_the_damn_walls(200, 5)
        self.walls_on_board['Vertical'] = self.create_the_damn_walls(5, 200)

        # Eeach player has these walls.
        self.wallsPlayer1 = list()
        self.wallsPlayer2 = list()
        i = 0
        while len(self.wallsPlayer1) < parameter.walls:
            # Initialize to False to avoid this warning: QGraphicsScene::addItem: item has already been added to this scene
            # In subsequent init calls with different wall numbers
            self.wallsPlayer1.append([QGraphicsRectItem(QRectF(-50 - i * 30, (parameter.size - 2) * 100, 5, 200)), False])
            self.wallsPlayer2.append([QGraphicsRectItem(QRectF((parameter.size * 100) + 50 + i * 30, 0, 5, 200)), False])

            if not self.wallsPlayer1[i][1]:
                self.scene.addItem(self.wallsPlayer1[i][0])
                self.scene.addItem(self.wallsPlayer2[i][0])
                self.wallsPlayer1[i][1] = True
                self.wallsPlayer2[i][1] = True
            i += 1

        # Board messages
        self.current_player.setPos(parameter.size * 100 + 25, 250)
        self.message.setPos(parameter.size * 100 + 25, 300)
        self.probablity.setPos(parameter.size * 100 + 25, 375)

        self.scene.addItem(self.player1)
        self.scene.addItem(self.player2)

        self.showMaximized()

    def create_the_damn_walls(self, width, height):
        """
        self.explanatory
            - return: List containing QGraphicsRectItem for every wall that will be used (player1 + player2)
        """
        lst = list()
        for i in range(parameter.walls * 2):
            wall = QGraphicsRectItem(QRectF(0, 0, width, height))
            # Red paint to symbolize my pain and suffering while implementing them
            wall.setBrush(QBrush(Qt.red))
            lst.append([wall, False])

        return lst

    def closeEvent(self, event):

        # Delete items in QGraphicsScene otherwise reducing the size of the board won't work
        for item in self.item_list:
            self.scene.removeItem(item)

        # Reset available walls
        for i in range(parameter.walls):
            if self.wallsPlayer1[i][1]:
                self.scene.removeItem(self.wallsPlayer1[i][0])
            if self.wallsPlayer2[i][1]:
                self.scene.removeItem(self.wallsPlayer2[i][0])
                self.wallsPlayer2[i][1] = True

        for i in range(len(self.walls_on_board['Horizontal'])):
            if self.walls_on_board['Horizontal'][i][1]:
                self.scene.removeItem(self.walls_on_board['Horizontal'][i][0])
            if self.walls_on_board['Vertical'][i][1]:
                self.scene.removeItem(self.walls_on_board['Vertical'][i][0])

        self.scene.removeItem(self.player1)
        self.scene.removeItem(self.player2)

        # Break from while loop in waitForKey()
        parameter.keyPressed = Qt.Key_Q
        parameter.running = False
        parameter.closed = True

    def keyPressEvent(self, event):

        parameter.keyPressed = event.key()
        # Break from waitForKey()
        parameter.running = False

    def display(self, lboard, msg=''):
        """
        Update board

                args:
                    - lboard: List containing the board state
        """
        white_player = lboard[0]
        black_player = lboard[1]
        horizontal_walls = lboard[2]
        vertical_walls = lboard[3]
        walls_left_p1 = lboard[4]
        walls_left_p2 = lboard[5]

        # Check diagonal jump
        if white_player[0] == black_player[0] and white_player[1] == black_player[1]:
            x = white_player[0] * 100 + 5 + 50
            x2 = white_player[0] * 100 - 50
            self.player1.setPos(x, (white_player[1] - (parameter.size - 1)) * -100 + 5)
            self.player2.setPos(x2, (white_player[1] - (parameter.size - 1)) * -100 + 5)

        else:
            self.player2.setPos(black_player[0] * 100 + 5, (black_player[1] - (parameter.size - 1)) * -100 + 5)
            self.player1.setPos(white_player[0] * 100 + 5, (white_player[1] - (parameter.size - 1)) * -100 + 5)

        self.addWallsBoard(horizontal_walls, self.walls_on_board['Horizontal'], 'Horizontal')
        self.addWallsBoard(vertical_walls, self.walls_on_board['Vertical'], 'Vertical')

        self.removeWalls(walls_left_p1, self.wallsPlayer1)
        self.removeWalls(walls_left_p2, self.wallsPlayer2)

    def removeWalls(self, wallsLeft, player):
        """
        Removes or adds available walls

                - args:
                    - wallsLeft: Walls each player has available
                    - player: Which player's walls to update
        """
        available_walls = 0
        i = 0
        found = False
        while i < parameter.walls and not found:
            if player[i][1]:
                available_walls += 1
            else:
                found = True
            i += 1

        if wallsLeft < available_walls:
            i = parameter.walls - 1
            found = False
            while i >= wallsLeft and not found:
                if player[i][1]:
                    self.scene.removeItem(player[i][0])
                    player[i][1] = False
                    found = True
                i -= 1

        elif wallsLeft > available_walls:
            # this could have been rewritten as a while loop, and stop once it updates the wall
            # in question (break) as it would've been more efficient. But, it wouldn't work in case
            # the game ends as it would only reset 1 wall instead of all of them.
            for i in range(parameter.walls):
                if not player[i][1]:
                    self.scene.addItem(player[i][0])
                    player[i][1] = True
                    if not parameter.endGame:
                        break

    def addWallsBoard(self, orientation_lst, walls_lst, orientation):
        """
        Calculates wall coordinates and updates their position on screen or removes the las placed wall
        when user switches between horizontal-vertical placements

            args:
                - orientation_lst: List containing the coordinates of the walls to update
                - walls_lst: List containing wall objects to place
                - orientation: Wall's orientation
        """
        # Hallelujah! This is the most efficient and consistent way I found to work.
        # Please forgive the usage of the 'forbidden instruction'
        i = 0
        while i < len(walls_lst):
            try:
                x = orientation_lst[i][0] * 100
                y = ((orientation_lst[i][1] - (parameter.size - 1)) * -100)

                # Offset depending on wall orientation
                x += 100 - 5/2 if orientation == 'Vertical' else 0
                y -= 5 / 2 if orientation == 'Horizontal' else 100

                walls_lst[i][0].setPos(x, y)
                if not walls_lst[i][1]:
                    walls_lst[i][1] = True  # Wall has been placed
                    self.scene.addItem(walls_lst[i][0])
                i += 1
            except IndexError:
                # Lenght of orientation_lst < self.walls_on_board['X']: Wall placement switched between vertical/horizontal or cancelled
                # Remove last placed wall from scene
                if walls_lst[i][1]:
                    walls_lst[i][1] = False
                    self.scene.removeItem(walls_lst[i][0])
                i += 1


parameter = Parameter()
mandatory_QApplication_creation = QApplication([])
super_awesome_board = Board()  # I ran out of variable name ideas
loading_bar = QProgressBar()
