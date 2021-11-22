# -*- coding: utf-8 -*-
"""Application graphical user interface.

This module manage the graphical user interface of the application.

"""

from PySide6 import QtWidgets
from PySide6.QtCore import QRectF, QPointF, Signal, QThread, Qt, QLine
from PySide6.QtGui import (
    QIcon,
    QAction,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QColorConstants,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QMenuBar,
    QStatusBar,
    QGridLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsSimpleTextItem,
    QGraphicsItem,
    QStyleOptionGraphicsItem,
    QWidget,
    QMenu,
    QMessageBox,
    QLabel,
    QLineEdit,
    QGraphicsPixmapItem,
)
import skimage.io
import numpy as np

from generator import Generator, SudokuDifficulty
from interfaces import AlgorithmType, Resolver


class DigitText(QGraphicsSimpleTextItem):
    def __init__(self, parent: QGraphicsItem = None):
        super().__init__(parent)

    def boundingRect(self):
        br = QGraphicsSimpleTextItem.boundingRect(self)
        return br.translated(-br.width() / 2, -br.height() / 2)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = 0
    ):
        painter.translate(
            -self.boundingRect().width() / 2, -self.boundingRect().height() / 2
        )
        QGraphicsSimpleTextItem.paint(self, painter, option, widget)


class MainWindow(QMainWindow):
    """
    A class to represent the app main window.
    """

    resolve = Signal((AlgorithmType, np.ndarray))
    analyse_picture = Signal(np.ndarray)

    def __init__(self, title: str, resolver: Resolver, importer):
        """
        Constructs all the necessary attributes for the main window object.
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon("./sudoku_csp/assets/icon.jpg"))

        self.layout = QGridLayout()
        self.sudoku_scene = QGraphicsScene()
        self.sudoku_view = QGraphicsView(self.sudoku_scene)
        self.picture_scene = QGraphicsScene()
        self.picture_view = QGraphicsView(self.picture_scene)
        self.layout.addWidget(self.sudoku_view, 0, 1)
        self.layout.addWidget(self.picture_view, 0, 0)

        self.generate_menu = QMenu("Generate", self)
        self.self_generate_menu = QMenu("Self generated puzzle", self)
        self.online_generate_menu = QMenu("Online generation (3x3 only)", self)
        self.solve_menu = QMenu("Solve", self)
        self.info_message = QLabel("", self)
        self.info_message.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        self.layout.addWidget(self.info_message, 1, 0)
        self.size_actions = list()
        self.picture = None

        self.size = 3
        self.length = self.size ** 2
        self.cell_width = 500 / self.length
        self.box_map = [[None for y in range(self.length)] for x in range(self.length)]
        self.digits_map = np.zeros((self.length, self.length), dtype=int)

        self.resolver_thread = QThread()
        self.resolve.connect(resolver.do_work)
        resolver.result_ready.connect(self.handle_result)
        resolver.error.connect(
            lambda x: self.handle_error("An error as occured while solving the puzzle.")
        )
        resolver.moveToThread(self.resolver_thread)

        self.importer_thread = QThread()
        self.analyse_picture.connect(importer.do_work)
        importer.result_ready.connect(self.handle_picture_import)
        importer.error.connect(
            lambda x: self.handle_error(
                "An error as occured while importing the puzzle."
            )
        )
        importer.moveToThread(self.importer_thread)

        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(self.layout)

        self.create_menus()
        self.create_sudoku_view(self.size)

        self.resolver_thread.start()
        self.importer_thread.start()

    def create_sudoku_view(self, n: int = 3):

        for y in range(0, n ** 2):
            for x in range(0, n ** 2):
                self.sudoku_scene.addRect(
                    QRectF(
                        self.cell_width * x,
                        self.cell_width * y,
                        self.cell_width,
                        self.cell_width,
                    )
                ),

                editor = QLineEdit()
                editor.setAlignment(Qt.AlignCenter)
                proxy = self.sudoku_scene.addWidget(editor)
                proxy.setPos(QPointF(self.cell_width * x + 1, self.cell_width * y + 1))
                proxy.setFont(QFont("Arial", self.cell_width / 2, QFont.Bold))
                proxy.setMinimumHeight(self.cell_width - 1)
                proxy.setMaximumWidth(self.cell_width - 1)

                self.box_map[x][y] = proxy

        # Draw visual lines
        pen = QPen()
        pen.setWidth(3)
        for i in range(1, n):
            self.sudoku_scene.addLine(
                i * n * self.cell_width,
                0,
                i * n * self.cell_width,
                n ** 2 * self.cell_width,
                pen,
            )
            self.sudoku_scene.addLine(
                0,
                i * n * self.cell_width,
                n ** 2 * self.cell_width,
                i * n * self.cell_width,
                pen,
            )

    def create_menus(self):
        self.setMenuBar(QMenuBar())

        import_action = QAction("Import", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.handle_import)

        self.menuBar().addAction(import_action)

        for menu, is_online in zip(
            [self.self_generate_menu, self.online_generate_menu], [False, True]
        ):

            generate_easy_action = QAction("Easy", self)
            generate_easy_action.setData(
                {"is_online": is_online, "difficulty": SudokuDifficulty.EASY}
            )
            generate_easy_action.setShortcut("Ctrl+E")
            generate_easy_action.setCheckable(False)
            generate_easy_action.triggered.connect(self.handle_generation)

            generate_medium_action = QAction("Medium", self)
            generate_medium_action.setData(
                {"is_online": is_online, "difficulty": SudokuDifficulty.MEDIUM}
            )
            generate_medium_action.setShortcut("Ctrl+M")
            generate_medium_action.setCheckable(False)
            generate_medium_action.triggered.connect(self.handle_generation)

            generate_hard_action = QAction("Hard", self)
            generate_hard_action.setData(
                {"is_online": is_online, "difficulty": SudokuDifficulty.HARD}
            )
            generate_hard_action.setShortcut("Ctrl+H")
            generate_hard_action.setCheckable(False)
            generate_hard_action.triggered.connect(self.handle_generation)

            menu.addActions(
                [
                    generate_easy_action,
                    generate_medium_action,
                    generate_hard_action,
                ]
            )

            self.generate_menu.addMenu(menu)

        self.menuBar().addMenu(self.generate_menu)

        size_menu = QMenu("Size", self)
        for i in np.arange(2, 7):
            action = QAction(f"{i}x{i}", self)
            action.setCheckable(True)
            action.setData(i)
            if i == self.size:
                action.setChecked(True)
            action.triggered.connect(self.handle_size_edit)
            self.size_actions.append(action)
            size_menu.addAction(action)

        self.generate_menu.addMenu(size_menu)

        solve_backtracking_action = QAction("Backtracking", self)
        solve_backtracking_action.triggered.connect(
            lambda x: self.handle_resolve(AlgorithmType.BACKTRACKING)
        )

        solve_mrv_action = QAction("MRV", self)
        solve_mrv_action.triggered.connect(
            lambda x: self.handle_resolve(AlgorithmType.MRV)
        )

        solve_ac3_action = QAction("AC-3", self)
        solve_ac3_action.triggered.connect(
            lambda x: self.handle_resolve(AlgorithmType.AC3)
        )

        solve_degree_h_action = QAction("Degree heuristic", self)
        solve_degree_h_action.triggered.connect(
            lambda x: self.handle_resolve(AlgorithmType.DEGREE_H)
        )

        solve_least_constraining_h_action = QAction("Least constraining value", self)
        solve_least_constraining_h_action.triggered.connect(
            lambda x: self.handle_resolve(AlgorithmType.LEAST_CONSTRAINING_H)
        )

        self.solve_menu.addActions(
            [
                solve_backtracking_action,
                solve_mrv_action,
                solve_ac3_action,
                solve_degree_h_action,
                solve_least_constraining_h_action,
            ]
        )
        self.menuBar().addMenu(self.solve_menu)

        self.setStatusBar(QStatusBar())

    def draw_number(self, number: int, pos: np.array):
        self.box_map[pos[0]][pos[1]].widget().setText(str(number))

    def clear_cell(self, pos: np.array):
        self.box_map[pos[0]][pos[1]].widget().clear()

    def update_sudoku_view(self):
        self.sudoku_scene.clear()

        self.create_sudoku_view(self.size)

        for y in range(self.length):
            for x in range(self.length):

                self.clear_cell([x, y])

                if self.digits_map[x, y] != 0:
                    self.draw_number(self.digits_map[x, y], np.array([x, y]))

    def handle_error(self, error_message: str):
        QMessageBox.critical(self, "Error", error_message)

    def handle_import(self):
        picture_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "/", "Image Files (*.png *.jpg *.bmp)"
        )
        if not picture_path:
            return
        self.picture = skimage.io.imread(picture_path)
        pixmap = QPixmap(picture_path)
        self.picture_scene.clear()
        self.picture_scene.addItem(QGraphicsPixmapItem(pixmap))
        self.picture_view.fitInView(
            self.picture_scene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self.analyse_picture.emit(self.picture)

    def handle_generation(self):
        action = self.sender()
        if action.data()["is_online"]:
            if self.size != 3:
                self.handle_error(
                    "Online generation of sudoku with size different than 3x3 is not currently supported."
                )
                return
            self.info_message.clear()
            self.digits_map = Generator.generate_online(
                self.size, action.data()["difficulty"]
            )
        else:
            self.info_message.setText(
                "The uniqueness of this homemade generated puzzle isn't guaranteed."
            )
            self.digits_map = Generator.generate_backtracking(
                self.size, action.data()["difficulty"]
            )

        self.update_sudoku_view()

    def handle_resolve(self, algorithm_type: AlgorithmType):
        print(f"Trying to resolve using {algorithm_type.value} algorithm...")
        for x in range(self.length):
            for y in range(self.length):
                if self.box_map[x][y].widget().text():
                    self.digits_map[x, y] = int(self.box_map[x][y].widget().text())
                else:
                    self.digits_map[x, y] = 0

        self.resolve.emit(algorithm_type, self.digits_map)

    def handle_picture_import(self, result: tuple):
        cols, rows = result
        pen = QPen()
        pen.setWidth(10)
        pen.setColor(QColorConstants.Red)

        for y in range(len(rows)):
            self.picture_scene.addLine(QLine(cols[0], rows[y], cols[-1], rows[y]), pen)

        for x in range(len(cols)):
            self.picture_scene.addLine(QLine(cols[x], rows[0], cols[x], rows[-1]), pen)

    def handle_result(self, algorithm_type: AlgorithmType, sudoku_map: np.array):
        print(f"Sudoku resolved using {algorithm_type.value} algorithm!")
        self.digits_map = sudoku_map
        self.update_sudoku_view()

    def handle_size_edit(self):
        if isinstance(self.sender(), QAction):
            for action in self.size_actions:
                if action != self.sender():
                    action.setChecked(False)

            if self.sender().data() != 3:
                self.online_generate_menu.setEnabled(False)
            else:
                self.online_generate_menu.setEnabled(True)

            self.size = int(self.sender().data())
            self.length = self.size ** 2
            self.cell_width = 500 / self.length

            self.sudoku_scene.clear()

            self.box_map = [
                [None for y in range(self.length)] for x in range(self.length)
            ]
            self.digits_map = np.zeros((self.length, self.length), dtype=int)

            self.update_sudoku_view()
