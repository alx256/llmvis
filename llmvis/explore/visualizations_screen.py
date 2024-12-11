from PyQt6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSizePolicy)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from llmvis.connections import OllamaConnection
from typing import Callable

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
        prompt_line_layout.addWidget(QLabel('Prompt: '))
        self.__prompt = QLineEdit()
        prompt_line_layout.addWidget(self.__prompt)
        prompt_line = QWidget()
        prompt_line.setLayout(prompt_line_layout)

        # Metric line with a dropdown for the metrics available
        metric_line_layout = QHBoxLayout()
        metric_line_layout.addWidget(QLabel('Metric: '))
        self.__metric_dropdown = QComboBox()
        self.__metric_dropdown.addItems(['Generation Shapley', 'Embedding Shapley'])
        metric_line_layout.addWidget(self.__metric_dropdown)
        metric_line = QWidget()
        metric_line.setLayout(metric_line_layout)

        # The start button, which starts the visualization
        self.__start_button = QPushButton('Start')
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

        self.__callback(self.__prompt.text(),
            self.__metric_dropdown.currentText())
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

        if data['service'] == 'Ollama':
            self.__conn = OllamaConnection(data['model'])

        layout = QVBoxLayout()
        title = QLabel('Select a visualization')
        title.setObjectName('title')
        layout.addWidget(title)

        token_importance_button = QPushButton('Word Importance')
        token_importance_button.clicked.connect(self.__token_importance)
        token_importance_button.setSizePolicy(QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed)

        layout.addWidget(token_importance_button)
        layout.addStretch()

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

        if metric == 'Generation Shapley':
            vis = self.__conn.word_importance_gen_shapley(prompt)
        elif metric == 'Embedding Shapley':
            vis = self.__conn.word_importance_embed_shapley(prompt)
        
        web_view = QWebEngineView()
        web_view.setHtml(vis.get_source())

        self.parentWidget().setCentralWidget(web_view)