'''
    File name: partie4.py
    Author: El Miri Hamza
    ID: 000479603
    Date: 07/04/2019

    Function:
        UI for main_partie4.py
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QGridLayout, QComboBox, QAction, QStatusBar, QSpinBox, QDoubleSpinBox, QPushButton, QFileDialog, QProgressBar
from config import parameter, super_awesome_board, loading_bar
from main_partie4 import *


class Main(QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.title = "Quoridor"
        self.x = 700
        self.y = 300
        self.width = 500
        self.height = 550
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Quoridor: Deluxe Edition')
        self.setFixedWidth(500)
        self.setFixedHeight(625)

        self.options = Options()

        self.setCentralWidget(self.options)
        self.show()

        menuBar = self.menuBar()
        file = menuBar.addMenu("File")
        file.addAction("Load AI")
        file.addAction("Save AI")
        file.triggered[QAction].connect(self.fileMenu)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # To be used in Buttons class
        self.loading_label = QLabel('')
        self.loading_bar_added = False

    def fileMenu(self, trigger):

        if trigger.text() == "Load AI":
            AI_load = QFileDialog.getOpenFileName(self, "Load AI", "", "Load AI file(*.npz)")
            data = np.load(AI_load[0])
            parameter.size = int(data['N'])
            parameter.walls = int(data['WALLS'])
            parameter.NN = (data['W1'], data['W2'])
            parameter.g_init = computeGraph()
            self.statusBar.showMessage(str(AI_load[0]) + '  Loaded', 5000)

        if trigger.text() == "Save AI":
            if parameter.NN is not None:
                AI_save = QFileDialog.getSaveFileName(self, "Save AI", "", "Save AI File(*.npz")
                np.savez(AI_save[0], N=parameter.size, WALLS=parameter.walls, W1=parameter.NN[0], W2=parameter.NN[1])
                self.statusBar.showMessage(str(AI_save[0]) + '  Saved', 5000)
            else:
                self.statusBar.showMessage("You must first create or load an AI first!", 5000)


class DropDownMenu(QComboBox):

    def __init__(self, menu, items):
        """
            - args:
                - menu: Menu text
                - items: Items to be added
        """
        super().__init__()

        for i in range(len(items)):
            self.addItem(items[i])

        self.menu = menu

        if menu == 'Strategy':
            self.currentTextChanged.connect(self.setStrategy)
        elif menu == 'Activation':
            self.currentTextChanged.connect(self.setActivation)
        else:
            self.currentIndexChanged.connect(self.epsMode)

    def setStrategy(self, text):
        parameter.learning_strategy = self.currentText()
    
    def setActivation(self, text):
        parameter.activation = self.currentText()
    
    def epsMode(self, text):
        parameter.epsilon_mode = self.currentText()


class Options(QWidget):
    """
    Option widgets
    """

    def __init__(self):
        super().__init__()

        self.grid = QGridLayout()

        # Create items
        self.strategy = DropDownMenu('Strategy', ['Q-Learning', 'Q-Lambda','DQ-Lambda','TD-Lambda'])
        self.activation = DropDownMenu('Activation', ['Sigmoid', 'ReLU', 'Swish'])
        self.epsilon_mode = DropDownMenu('Eps mode', ['Static', 'Adaptive'])

        self.board_size = self.create_box('BoardSize', 'SpinBox', [3, 9], 1)
        self.nbr_walls = self.create_box('Walls', 'SpinBox', [0, 10], 1)
        self.nbr_games_train = self.create_box('Train', 'SpinBox', [1, 10000000], 500)
        self.nbr_games_compare = self.create_box('Compare', 'SpinBox', [1, 10000000], 500)
        self.nbr_neurons = self.create_box('Neurons', 'SpinBox', [0, 10000000], 1)
        self.epsilon = self.create_box('Epsilon', 'DoubleSpinBox', [0.00, 1.00], 0.05)
        self.alpha = self.create_box('Alpha', 'DoubleSpinBox', [0.00, 1.00], 0.05)
        self.lambda_ = self.create_box('Lambda', 'DoubleSpinBox', [0.00, 1.00], 0.05)
        self.sigma = self.create_box('Sigma', 'DoubleSpinBox', [0, 1], 0.01)
        
        

        self.trainAI = Buttons('Train', 'Train AI')
        self.compareAI = Buttons('Compare', 'Compare AI')
        self.playAI = Buttons('PlayAI', 'Play AI')
        self.playHuman = Buttons('PlayHuman', 'Play Human')

        self.initWidgets()

    def initWidgets(self):
        # Place items in a grid

        labels = ['Strategy:', 'Activation function:','Epsilon mode:','Neurons', 'Board Size:', 'Number of walls:',
                  'Number of games to train the AI:', 'Number of games to compare the AI:',
                  'Epsilon:', 'Epsilon noise (sigma)', 'Learning rate:', 'Lambda']

        options = [self.strategy, self.activation, self.epsilon_mode, self.nbr_neurons, self.board_size,
                   self.nbr_walls, self.nbr_games_train, self.nbr_games_compare, self.epsilon,
                   self.sigma, self.alpha, self.lambda_, self.trainAI, self.compareAI, self.playAI,
                   self.playHuman]

        for i in range(len(options)):
            if i < len(labels):
                self.grid.addWidget(QLabel(labels[i]), i, 0)
                self.grid.addWidget(options[i], i, 1)
            else:
                if i % 2 == 0:
                    self.grid.addWidget(options[i], i, 0)
                else:
                    self.grid.addWidget(options[i], i - 1, 1)

        self.setLayout(self.grid)

    def create_box(self, option, box_type, box_range, step):
        """
        Creates SpinBoxes

                args:
                    - option: String with the name of the option to be created
                    - box_type: Either SpinBox or DoubleSpinBox
                    - box_rage: Minimum and maximum values of the box
                    - step: Increment size
        """

        if box_type == 'SpinBox':
            box = QSpinBox()
        else:
            box = QDoubleSpinBox()

        box.setSingleStep(step)
        box.setMinimum(box_range[0])
        box.setMaximum(box_range[1])

        # Set values

        if option == 'BoardSize':
            box.valueChanged.connect(self.update_board_size)
            box.setValue(parameter.size)
        elif option == 'Walls':
            box.valueChanged.connect(self.update_walls)
            box.setValue(parameter.walls)
        elif option == 'Epsilon':
            box.valueChanged.connect(self.update_epsilon)
            box.setValue(parameter.epsilon)
        elif option == 'Alpha':
            box.valueChanged.connect(self.update_alpha)
            box.setValue(parameter.alpha)
        elif option == 'Lambda':
            box.valueChanged.connect(self.update_lambda)
            box.setValue(parameter.lamb)
        elif option == 'Neurons':
            box.valueChanged.connect(self.update_neurons)
            box.setValue(parameter.neurons)
        elif option == 'Train':
            box.valueChanged.connect(self.update_train)
            box.setValue(parameter.train)
        elif option == 'Compare':
            box.valueChanged.connect(self.update_compare)
            box.setValue(parameter.compare)
        elif option == 'Decreasing Epsilon':
            box.valueChanged.connect(self.update_decrease_eps)
            box.setValue(parameter.eps_decrease)
        elif option == 'Epsilon end value':
            box.valueChanged.connect(self.update_eps_end_value)
        elif option == 'Sigma':
            box.valueChanged.connect(self.update_sigma)

        return box

    def update_sigma(self, value):
        parameter.sigma = value

    def update_eps_end_value(self, value):
        parameter.eps_end_value = value
        self.update_decrease_eps()

    def update_decrease_eps(self):
        parameter.eps_decrease = (parameter.epsilon - parameter.eps_end_value) / parameter.train
        
    def update_board_size(self, value):
        parameter.size = value

    def update_walls(self, value):
        parameter.walls = value

    def update_neurons(self, value):
        parameter.neurons = value

    def update_epsilon(self, value):
        parameter.epsilon = value
        self.update_decrease_eps()

    def update_alpha(self, value):
        parameter.alpha = value

    def update_lambda(self, value):
        parameter.lamb = value

    def update_train(self, value):
        parameter.train = value
        self.update_decrease_eps()

    def update_compare(self, value):
        parameter.compare = value


class Buttons(QPushButton):
    def __init__(self, button, text):
        """
                - args:
                    - button: Name of the button to be created
                    - text: Button text
        """
        super().__init__()
        self.setText(text)

        if button == 'Train':
            self.clicked.connect(self.trainAI)
        elif button == 'Compare':
            self.clicked.connect(self.compareAI)
        elif button == 'PlayAI':
            self.clicked.connect(self.playAI)
        else:
            self.clicked.connect(self.playHuman)


    def trainAI(self):
        parameter.training = True
        parameter.board = startingBoard()
        parameter.g_init = computeGraph()
        parameter.g = parameter.g_init.copy()
        parameter.NN = createNN(2 * parameter.size ** 2 + 2 * (parameter.size - 1) ** 2 + 2 * (parameter.walls + 1), parameter.neurons)

        if not main.loading_bar_added:
            main.statusBar.insertWidget(0, loading_bar, 20)
            main.loading_bar_added = True

        loading_bar.setValue(0)

        # Remove previous Qlabel before adding the new one
        main.statusBar.removeWidget(main.loading_label)
        main.loading_label = QLabel("Training hard. Please wait...")
        main.statusBar.insertWidget(0, main.loading_label, 10)
        train(parameter.NN, parameter.train)
        loading_bar.setValue(100)  # Because it stops at 99% >:c
        main.statusBar.insertWidget(0, main.loading_label, 10)

        # For some reason, showMessage doesn't obscure the widgets as stated  in the documentation
        # Only way to do it is by main.statusBar.removeWidget(main.loading_bar) and then using showMessage
        # to display it. But nothing is that simple! Because for some reason, once main.loading_bar is removed
        # from statusBar, it can no longer be added. So, I'm using QLabels to get around the problem.

        # main.statusBar.removeWidget(main.oading_bar)
        # main.statusBar.showMessage("Ready to kick some butt!", 5000)
        main.statusBar.removeWidget(main.loading_label)
        main.loading_label = QLabel("Ready to kick some butt!")
        main.statusBar.insertWidget(0, main.loading_label, 10)

    def compareAI(self):
        AI_load = QFileDialog.getOpenFileName(self, "Load AI", "", "Load AI file(*.npz)")
        filename = AI_load[0]

        if not main.loading_bar_added:
            main.statusBar.insertWidget(0, loading_bar, 20)
            main.loading_bar_added = True
        loading_bar.setValue(0)

        main.statusBar.removeWidget(main.loading_label)

        main.loading_label = QLabel("{0} Games with Epsilon = {1}".format(parameter.compare, parameter.epsilon))
        main.statusBar.insertWidget(0, main.loading_label, 10)
        wins = compare(parameter.NN, filename, parameter.compare, parameter.epsilon)
        loading_bar.setValue(100)

        # main.statusBar.removeWidget(loading_bar)
        # main.statusBar.showMessage("Games won by the AI:  " + wins, 5000)
        main.statusBar.removeWidget(main.loading_label)
        main.loading_label = QLabel("Games won by the AI:  " + wins + " ")
        main.statusBar.insertWidget(0, main.loading_label, 10)

    def playAI(self):
        super_awesome_board.initBoard()
        human = Player_Human('Humain')
        agent = Player_AI(parameter.NN, 0.0, None, 'IA')
        parameter.training = False
        parameter.boardClosed = False
        play(human, agent)

    def playHuman(self):
        parameter.training = False
        parameter.boardClosed = False

        super_awesome_board.initBoard()
        old_N = parameter.size
        old_walls = parameter.walls
        parameter.g_init = computeGraph()
        human1 = Player_Human('Humain 1')
        human2 = Player_Human('Humain 2')

        play(human1, human2)
        parameter.size = old_N
        parameter.walls = old_walls
        parameter.g_init = computeGraph()


if __name__ == '__main__':
    ui = QApplication(sys.argv)
    main = Main()
    sys.exit(ui.exec_())
