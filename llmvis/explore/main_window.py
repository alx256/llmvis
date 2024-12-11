from PyQt6.QtWidgets import QMainWindow
from llmvis.explore.home_screen import HomeScreen

class MainWindow(QMainWindow):
    """
    The main window for the LLMVis Explore application, containing the
    Home Screen by default.
    """

    def __init__(self):
        super().__init__()

        # 720p size
        self.resize(1280, 720)
        self.setWindowTitle('LLMVis Explore')
        # Start with the home screen
        self.setCentralWidget(HomeScreen())