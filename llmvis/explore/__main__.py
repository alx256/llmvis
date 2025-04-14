from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase
import sys
import warnings

from llmvis.explore.main_window import MainWindow
from llmvis.visualization.linked_files import absolute_path

app = QApplication(sys.argv)

# Load custom font to match visualizations
font_file_path = absolute_path("assets/fonts/DidactGothic-Regular.ttf")
id = QFontDatabase.addApplicationFont(str(font_file_path))

if id < 0:
    warnings.warn("Failed to load custom fonts")

app.setStyleSheet(
    """
* {
    font-family: "Didact Gothic";
    font-size: 18px;
    color: rgb(191, 191, 191);
    background-color: rgb(23, 21, 33);
}

QLabel#title {
    font-size: 45px;
}

QPushButton {
    border: 2px solid #5A9;
    border-radius: 15px;
    background-color: transparent;
    color: white;
    padding: 5px 15px;
}
"""
)

window = MainWindow()
window.show()

app.exec()
