from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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
from llmvis import visualization
from llmvis.connections import (
    ImportanceCalculation,
    ImportanceMetric,
    OllamaConnection,
    UnitType,
    WatsonXConnection,
)
from typing import Callable, Optional

from llmvis.visualization import Visualizer
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
        self.clicked.connect(onclick)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class VisualizationCreationDialog(QDialog):
    """
    Dialog shown where the users can provide the parameters
    for the word importance visualizations.
    """

    def __init__(self, callback: Callable, visualization_name: str):
        """
        Create a new `VisualizationCreationDialog` with a callback function.

        Args:
            callback (Callable): Callback function that is called
                with a single argument containing a dictionary with
                all the properties.
        """

        super().__init__()

        self.setWindowTitle(f"Create new {visualization_name} visualization")

        self.__layout__ = QVBoxLayout()
        self.setLayout(self.__layout__)

        self.__callback = callback
        self.__properties__ = {}

    def exec(self):
        """
        Execute this `VisualizationCreationDialog`, showing it to the user.
        """

        # The start button, which starts the visualization
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.__start)
        self.__layout__.addWidget(start_button)

        return super().exec()

    def add_line_edit(self, name: str, identifier: str, default: str = ""):
        """
        Add a new line edit to this dialog.

        Args:
            name (str): The name that should be shown before this line edit.
            identifier (str): The identifier that will be used for storing
                the `QLineEdit` for use later.
            default (str): The default text that this line edit should show.
        """

        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel(f"{name}: "))
        line_edit = QLineEdit()
        line_edit.setText(default)
        self.__properties__[identifier] = line_edit
        line_layout.addWidget(self.__properties__[identifier])
        line = QWidget()
        line.setLayout(line_layout)
        self.__layout__.addWidget(line)

    def add_dropdown(self, name: str, options: list[str], identifier: str):
        """
        Add a new dropdown menu to this dialog.

        Args:
            name (str): The name that should be shown before this dropdown
                menu.
            options (list[str]): The different available options.
            identifier (str): The identifier that will be used for storing
                the `QComboBox` for use later.
        """

        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel(f"{name}: "))
        combo_box = QComboBox()
        combo_box.addItems(options)
        self.__properties__[identifier] = combo_box
        line_layout.addWidget(self.__properties__[identifier])
        line = QWidget()
        line.setLayout(line_layout)
        self.__layout__.addWidget(line)

    def add_toggle(self, name: str, identifier: str):
        """
        Add a new togglable check box to this dialog.

        Args:
            name (str): The name that should be shown before this
                togglable check box.
            identifier (str): The identifier that will be used for
                storing the `QCheckBox` for use later.
        """

        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel(f"{name}: "))
        check_box = QCheckBox()
        self.__properties__[identifier] = check_box
        line_layout.addWidget(self.__properties__[identifier])
        line = QWidget()
        line.setLayout(line_layout)
        self.__layout__.addWidget(line)

    def __start(self):
        """
        Called when the 'start' button is pressed. Calls the callback
        function with the properties dictionary as an argument,
        closing this box in the process.
        """

        self.__callback(self.__properties__)
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
            self.__unit_importance__,
        )
        grid_layout.addWidget(token_importance_button, 0, 0)

        temperature_impact_button = VisualizationButton(
            "Temperature Impact",
            "See how the temperature parameter impacts model performance",
            "explore/assets/temperature_impact_icon.png",
            self.__temperature_impact__,
        )
        grid_layout.addWidget(temperature_impact_button, 0, 1)

        sandbox_button = VisualizationButton(
            "Sandbox",
            "See how model outputs are generated by using your own settings",
            "explore/assets/sandbox_icon.png",
            self.__sandbox__,
        )
        grid_layout.addWidget(sandbox_button, 1, 0)

        grid.setLayout(grid_layout)
        layout.addWidget(grid)

        self.setLayout(layout)

    def __unit_importance__(self):
        """
        Called when the option for seeing the unit importance visualization
        is selected
        """

        dialog = VisualizationCreationDialog(
            callback=self.__start_unit_importance__,
            visualization_name="unit importance",
        )
        dialog.add_line_edit("Prompt", "prompt")
        dialog.add_line_edit("System Prompt", "system_prompt")
        dialog.add_dropdown(
            "Importance Metric", ["Inverse Cosine", "Shapley"], "importance_metric"
        )
        dialog.add_dropdown("Calculation", ["Generation", "Embedding"], "calculation")
        dialog.add_dropdown(
            "Unit Type", ["Segment", "Word", "Sentence", "Token"], "unit_type"
        )
        dialog.add_line_edit("Sampling Ratio", "sampling_ratio")
        dialog.add_toggle(
            "Calculate Perplexity Difference", "calculate_perplexity_difference"
        )
        dialog.add_toggle("Test System Prompt", "test_system_prompt")
        dialog.add_line_edit("Similarity Threshold", "similarity_threshold")
        dialog.exec()

    def __temperature_impact__(self):
        """
        Called when the option for seeing the temperature impact visualization
        is selected
        """

        dialog = VisualizationCreationDialog(
            callback=self.__start_temperature_impact__,
            visualization_name="temperature impact",
        )
        dialog.add_line_edit("Prompt", "prompt")
        dialog.add_line_edit("k", "k", default="5")
        dialog.add_line_edit("System Prompt", "system_prompt")
        dialog.add_line_edit("Start", "start", default="0.0")
        dialog.add_line_edit("End", "end", default="1.0")
        dialog.add_toggle("Show alternative tokens", "alternative_tokens")
        dialog.exec()

    def __sandbox__(self):
        """
        Called when the option for seeing the sandbox visualization
        is selected
        """

        dialog = VisualizationCreationDialog(
            callback=self.__start_sandbox__, visualization_name="sandbox"
        )
        dialog.add_line_edit("Prompt", "prompt")
        dialog.add_line_edit("System Prompt", "system_prompt")
        dialog.add_line_edit("Temperature", "temperature", default="0.7")
        dialog.exec()

    def __start_unit_importance__(self, properties: dict[str, QWidget]):
        """
        Callback function for starting the unit importance visualization.

        Args:
            properties (dict[str, QWidget]): The properties passed when the
                callback function is called.
        """

        prompt = properties["prompt"].text()
        system_prompt = properties["system_prompt"].text()
        system_prompt = None if len(system_prompt) == 0 else system_prompt
        importance_metric_selected = properties["importance_metric"].currentText()

        if importance_metric_selected == "Inverse Cosine":
            importance_metric = ImportanceMetric.INVERSE_COSINE
        elif importance_metric_selected == "Shapley":
            importance_metric = ImportanceMetric.SHAPLEY

        calculation_selected = properties["calculation"].currentText()

        if calculation_selected == "Generation":
            calculation = ImportanceCalculation.GENERATION
        elif calculation_selected == "Embedding":
            calculation = ImportanceCalculation.EMBEDDING

        unit_type_selected = properties["unit_type"].currentText()

        if unit_type_selected == "Segment":
            unit_type = UnitType.SEGMENT
        elif unit_type_selected == "Word":
            unit_type = UnitType.WORD
        elif unit_type_selected == "Sentence":
            unit_type = UnitType.SENTENCE
        elif unit_type_selected == "Token":
            unit_type = UnitType.TOKEN

        sampling_ratio = properties["sampling_ratio"].text()
        sampling_ratio = 0.0 if len(sampling_ratio) == 0 else float(sampling_ratio)
        use_perplexity_difference = properties[
            "calculate_perplexity_difference"
        ].isChecked()
        test_system_prompt = properties["test_system_prompt"].isChecked()
        similarity_threshold = properties["similarity_threshold"].text()
        similarity_threshold = (
            0.1 if len(similarity_threshold) == 0 else float(similarity_threshold)
        )

        vis = self.__conn.unit_importance(
            prompt,
            system_prompt,
            importance_metric,
            calculation,
            unit_type,
            sampling_ratio,
            use_perplexity_difference,
            test_system_prompt,
            similarity_threshold,
        )
        self.__start__(vis)

    def __start_temperature_impact__(self, properties: dict[str, QWidget]):
        """
        Callback function for starting the temperature impact visualization.

        Args:
            properties (dict[str, QWidget]): The properties passed when the
                callback function is called.
        """

        prompt = properties["prompt"].text()
        k = int(properties["k"].text())
        system_prompt = properties["system_prompt"].text()
        system_prompt = None if len(system_prompt) == 0 else system_prompt
        start = float(properties["start"].text())
        end = float(properties["end"].text())
        alternative_tokens = properties["alternative_tokens"].isChecked()

        vis = self.__conn.k_temperature_sampling(
            prompt, k, system_prompt, start, end, alternative_tokens
        )
        self.__start__(vis)

    def __start_sandbox__(self, properties: dict[str, QWidget]):
        """
        Callback function for starting the sandbox visualization.

        Args:
            properties (dict[str, QWidget]): The properties passed when the
                callback function is called.
        """

        prompt = properties["prompt"].text()
        system_prompt = properties["system_prompt"].text()
        system_prompt = None if len(system_prompt) == 0 else system_prompt
        temperature = float(properties["temperature"].text())
        vis = self.__conn.sandbox(prompt, system_prompt, temperature)
        self.__start__(vis)

    def __start__(self, vis: Visualizer):
        """
        Render a given visualization in this window.

        Args:
            vis (Visualizer): The visualization that should be rendered
        """

        web_view = QWebEngineView()
        web_view.setHtml(vis.get_source())
        self.parentWidget().setCentralWidget(web_view)
