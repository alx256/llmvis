from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSizePolicy,
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from llmvis.connections import OllamaConnection, WatsonXConnection
from typing import Callable, Optional

from llmvis.visualization.linked_files import absolute_path


class VisualizationIcon(QLabel):
    """
    An icon for a visualization. Tries to retain aspect ratio.
    """

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.__original_pixmap__ = pixmap
        self.setPixmap(pixmap)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event):
        if self.pixmap().isNull():
            # Prevent warning
            return

        # Scale pixmap so that it fills the label, retaining the aspect ratio
        self.setPixmap(
            self.__original_pixmap__.scaled(
                self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )


class VisualizationButton(QPushButton):
    """
    A button to open a visualization. Shows an icon, some text for the name,
    and some smaller text describing what the visualization actually is.
    """

    def __init__(self, text: str, subtext: str, image_path: str, onclick: Callable):
        """
        Create a new `VisualizationButton`.

        Args:
            text (str): The text for the larger of the two labels.
            subtext (str): The text for the smaller of the two labels.
            image_path (str): The string (i.e. not `Path` object) path for
                where the image that should be used for this button's icon is
                stored.
            onclick (Callable): A `Callable` that should be called when this
                button is pressed.
        """

        super().__init__()

        layout = QVBoxLayout()
        image_pixmap = QPixmap(str(absolute_path(str(image_path))))
        icon = VisualizationIcon(pixmap=image_pixmap, parent=self)
        text_label = QLabel(text)
        text_label.setObjectName("visualizationText")
        text_label.setWordWrap(True)
        subtext_label = QLabel(subtext)
        subtext_label.setObjectName("visualizationSubtext")
        subtext_label.setWordWrap(True)
        layout.addWidget(icon)
        layout.addWidget(text_label)
        layout.addWidget(subtext_label)

        self.setLayout(layout)
        self.mouseReleaseEvent = onclick
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class WordImportanceDialog(QDialog):
    """
    Dialog shown where the users can provide the parameters
    for the word importance visualizations.
    """

    def __init__(self, callback: Callable):
        """
        Create a new `WordImportanceDialog` with a callback function.

        Args:
            callback (Callable): Callback function that is called
                with argument 1 being the prompt the user wishes to
                use and argument 2 being the name of the metric that
                should be used.
        """

        super().__init__()

        self.setWindowTitle("Create new word importance visualization")

        layout = QVBoxLayout()

        self.__callback = callback

        # Prompt line with a text input for the prompt
        prompt_line_layout = QHBoxLayout()
        prompt_line_layout.addWidget(QLabel("Prompt: "))
        self.__prompt = QLineEdit()
        prompt_line_layout.addWidget(self.__prompt)
        prompt_line = QWidget()
        prompt_line.setLayout(prompt_line_layout)

        # Metric line with a dropdown for the metrics available
        metric_line_layout = QHBoxLayout()
        metric_line_layout.addWidget(QLabel("Metric: "))
        self.__metric_dropdown = QComboBox()
        self.__metric_dropdown.addItems(["Generation Shapley", "Embedding Shapley"])
        metric_line_layout.addWidget(self.__metric_dropdown)
        metric_line = QWidget()
        metric_line.setLayout(metric_line_layout)

        # The start button, which starts the visualization
        self.__start_button = QPushButton("Start")
        self.__start_button.clicked.connect(self.__start)

        layout.addWidget(prompt_line)
        layout.addWidget(metric_line)
        layout.addWidget(self.__start_button)

        self.setLayout(layout)

    def __start(self):
        """
        Called when the 'start' button is pressed. Calls the callback
        function with the prompt and the metric name as arguments,
        closing this box in the process.
        """

        self.__callback(self.__prompt.text(), self.__metric_dropdown.currentText())
        self.close()


class VisualizationsScreen(QWidget):
    """
    The screen where the user can select which kind of visualization
    they wish to see.
    """

    def __init__(self, data: dict[str, str]):
        """
        Create a new `VisualizationScreen`.
        """

        super().__init__()

        if data["service"] == "Ollama":
            self.__conn = OllamaConnection(data["model"])
        elif data["service"] == "watsonx.ai":
            self.__conn = WatsonXConnection(
                data["api_key"], data["project_id"], data["model"], data["location"]
            )

        layout = QVBoxLayout()
        title = QLabel("What do you want to explore?")
        title.setObjectName("title")
        layout.addWidget(title)

        grid = QWidget()
        grid_layout = QGridLayout()

        token_importance_button = VisualizationButton(
            "Unit Importance",
            "See which parts of a prompt can be removed",
            "explore/assets/unit_importance_icon.png",
            self.__token_importance,
        )
        grid_layout.addWidget(token_importance_button, 0, 0)

        temperature_impact_button = VisualizationButton(
            "Temperature Impact",
            "See how the temperature parameter impacts model performance",
            "explore/assets/temperature_impact_icon.png",
            None,
        )
        grid_layout.addWidget(temperature_impact_button, 0, 1)

        sandbox_button = VisualizationButton(
            "Sandbox",
            "See how model outputs are generated by using your own settings",
            "explore/assets/sandbox_icon.png",
            None,
        )
        grid_layout.addWidget(sandbox_button, 1, 0)

        grid.setLayout(grid_layout)
        layout.addWidget(grid)

        self.setLayout(layout)

    def __token_importance(self):
        """
        Called when the option for seeing the token importance visualization
        is selected
        """

        dialog = WordImportanceDialog(self.__start_visualization)
        dialog.exec()

    def __start_visualization(self, prompt, metric):
        """
        Start the visualization for a given prompt and metric.

        Args:
            prompt (str): The prompt that the visualization should be
                shown for.
            metric (str): The name of the metric that should be
                visualized.
        """

        if metric == "Generation Shapley":
            vis = self.__conn.unit_importance(prompt)
        elif metric == "Embedding Shapley":
            vis = self.__conn.unit_importance(prompt)

        web_view = QWebEngineView()
        web_view.setHtml(vis.get_source())

        self.parentWidget().setCentralWidget(web_view)
