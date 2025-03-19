from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QPushButton,
    QDialog,
    QLineEdit,
    QComboBox,
    QSizePolicy,
    QListView,
)
import ollama
import json
from pathlib import Path
import os
from typing import Callable

import requests

from llmvis.explore.visualizations_screen import VisualizationsScreen
from llmvis.core.linked_files import absolute_path

SAVED_CONNECTIONS_DIR = absolute_path("__saved_connections")


class NewConnectionDialog(QDialog):
    """
    Dialog that is shown to allow the user to provide the necessary details
    to create a new connection (i.e name, service and model).
    """

    def __init__(self, callback: Callable):
        """
        Create a new `NewConnectionDialog` with a provided callback function.

        Args:
            callback (Callable): Function to be called when this dialog has
                been closed.
        """

        super().__init__()

        self.__ollama_models__ = None
        self.__watsonx_ai_models__ = None
        self.__callback = callback
        self.setWindowTitle("Create new connection")

        # Line with "Enter name: " and a textbox input
        name_line_layout = QHBoxLayout()
        name_line_layout.addWidget(QLabel("Enter name: "))
        self.__name_field = QLineEdit()
        name_line_layout.addWidget(self.__name_field)
        name_line = QWidget()
        name_line.setLayout(name_line_layout)

        services = ["Ollama", "watsonx.ai"]

        # Line with "Select model: " and a drop down
        model_line_layout = QHBoxLayout()
        model_line_layout.addWidget(QLabel("Select model: "))
        self.__models_dropdown = QComboBox()
        self.__models_dropdown.addItems(self.__get_models(services[0]))
        model_line_layout.addWidget(self.__models_dropdown)
        model_line = QWidget()
        model_line.setLayout(model_line_layout)

        # Line with "Select service: " and a dropdown
        service_line_layout = QHBoxLayout()
        service_line_layout.addWidget(QLabel("Select service: "))
        self.__services_dropdown = QComboBox()
        self.__services_dropdown.addItems(services)
        self.__services_dropdown.currentTextChanged.connect(self.__update_service)
        service_line_layout.addWidget(self.__services_dropdown)
        service_line = QWidget()
        service_line.setLayout(service_line_layout)

        self.__extra_items__ = QWidget()
        self.__extra_items__.setLayout(QVBoxLayout())
        self.__api_key_line_edit__ = None
        self.__project_id_line_edit__ = None
        self.__location_row_dropdown__ = None

        submit_button = QPushButton(text="Create")
        submit_button.clicked.connect(self.__submit)

        # Label used to show any errors that occur
        self.__errors_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(name_line)
        layout.addWidget(service_line)
        layout.addWidget(model_line)
        layout.addWidget(self.__extra_items__)
        layout.addWidget(submit_button)
        layout.addWidget(self.__errors_label)

        self.setLayout(layout)

    def __update_service(self, service_name: str):
        """
        Called when a new service has been selected in the dropdown.
        Update the model drop down to reflect the new service and the
        models available with it.

        Args:
            service_name (str): The name of the service that has been
                selected by the user.
        """

        self.__models_dropdown.clear()
        self.__models_dropdown.addItems(self.__get_models(service_name))

        # Clear extra items
        for i in reversed(range(self.__extra_items__.layout().count())):
            widget = self.__extra_items__.layout().takeAt(i).widget()

            if widget is not None:
                widget.deleteLater()

        if service_name == "watsonx.ai":
            api_key_row = QWidget()
            api_key_row.setLayout(QHBoxLayout())
            api_key_row.layout().addWidget(QLabel("Enter watsonx.ai API Key: "))
            self.__api_key_line_edit__ = QLineEdit()
            api_key_row.layout().addWidget(self.__api_key_line_edit__)
            self.__extra_items__.layout().addWidget(api_key_row)

            project_id_row = QWidget()
            project_id_row.setLayout(QHBoxLayout())
            project_id_row.layout().addWidget(QLabel("Enter Project ID: "))
            self.__project_id_line_edit__ = QLineEdit()
            project_id_row.layout().addWidget(self.__project_id_line_edit__)
            self.__extra_items__.layout().addWidget(project_id_row)

            location_row = QWidget()
            location_row.setLayout(QHBoxLayout())
            location_row.layout().addWidget(QLabel("Enter Location: "))
            self.__location_row_dropdown__ = QComboBox()
            self.__location_row_dropdown__.addItem("us-south")
            self.__location_row_dropdown__.addItem("eu-de")
            self.__location_row_dropdown__.addItem("eu-gb")
            self.__location_row_dropdown__.addItem("jp-tok")
            self.__location_row_dropdown__.addItem("au-syd")
            self.__location_row_dropdown__.addItem("ca-tor")
            location_row.layout().addWidget(self.__location_row_dropdown__)
            self.__extra_items__.layout().addWidget(location_row)

    def __get_models(self, service: str) -> list[str]:
        """
        Get the list of models for a specified service.

        Args:
            service (str): The service that the model list
                should be requested from.

        Returns:
            A list of strings containing the identifiers of each
                of the models that are available.
        """

        if service == "Ollama":
            return self.__get_ollama_models()
        elif service == "watsonx.ai":
            return self.__get_watsonx_ai_models__()

        return []

    def __get_ollama_models(self) -> list[str]:
        """
        Get the list of models for an Ollama service.

        Returns:
            A list of strings of models installed on the
                user's device under Ollama.
        """
        if self.__ollama_models__ is not None:
            return self.__ollama_models__

        response = ollama.list()
        self.__ollama_models__ = [model.model for model in response.models]

        return self.__ollama_models__

    def __get_watsonx_ai_models__(self) -> list[str]:
        """
        Get the list of models for a watsonx.ai service.

        Returns:
            A list of strings of models available on the
                remote watsonx.ai service.
        """

        if self.__watsonx_ai_models__ is not None:
            return self.__watsonx_ai_models__

        response = requests.get(
            "https://eu-de.ml.cloud.ibm.com/ml/v1/foundation_model_specs?version=2024-03-14"
        )
        response_json = response.json()
        self.__watsonx_ai_models__ = []

        for resource in response_json["resources"]:
            self.__watsonx_ai_models__.append(resource["model_id"])

        return self.__watsonx_ai_models__

    def __submit(self):
        """
        Called when the submit button is pressed on the dialog.
        Checks that there are no missing or invalid fields and
        saves the connection locally to the user's device. This
        also calls the callback function provided in the
        constructor.
        """

        self.__errors_label.clear()
        errors = ""

        # Check that the name field is not blank
        if self.__name_field.text() == "":
            errors += "- Name cannot be blank!\n"

        if (
            self.__api_key_line_edit__ is not None
            and self.__api_key_line_edit__.text() == ""
        ):
            errors += "- API Key cannot be blank!\n"

        if (
            self.__project_id_line_edit__ is not None
            and self.__project_id_line_edit__.text() == ""
        ):
            errors += "- Project ID cannot be blank!\n"

        if errors != "":
            self.__errors_label.setText(errors)
            return

        # Save this to a JSON
        connection = {
            "service": self.__services_dropdown.currentText(),
            "model": self.__models_dropdown.currentText(),
        }

        if self.__api_key_line_edit__ is not None:
            connection["api_key"] = self.__api_key_line_edit__.text()

        if self.__project_id_line_edit__ is not None:
            connection["project_id"] = self.__project_id_line_edit__.text()

        if self.__location_row_dropdown__ is not None:
            connection["location"] = self.__location_row_dropdown__.currentText()

        # Create data directory if it does not exist
        data_dir = SAVED_CONNECTIONS_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        file = (data_dir / (self.__name_field.text() + ".json")).open("w")
        file.write(json.dumps(connection))
        file.close()

        self.close()
        self.__callback()


class HomeScreen(QWidget):
    """
    Widget for the home screen, containing some welcome labels
    and buttons for creating new connections or using previously
    made ones.
    """

    def __init__(self):
        """
        Create a new `HomeScreen`.
        """

        super().__init__()

        layout = QVBoxLayout()
        title = QLabel("Welcome to LLMVis Explore")
        title.setObjectName("title")
        layout.addWidget(title)

        welcome_text = QLabel(
            """
        <p>LLMVis Explore allows you to experiment with the different
        metrics available in LLMVis.</p>

        <p>To begin, create a new connection to an LLM hosting service
        or use an existing one.</p>
        """
        )
        layout.addWidget(welcome_text)

        self.__connections_list_layout = QVBoxLayout()
        self.__populate_connections_list()

        connections_list = QWidget()
        connections_list.setLayout(self.__connections_list_layout)
        layout.addWidget(connections_list)

        new_connection_button = QPushButton(text="New connection...")
        new_connection_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        new_connection_button.clicked.connect(self.__show_new_connection_dialog)

        layout.addWidget(new_connection_button)
        layout.addStretch()

        layout.setSpacing(45)

        self.setLayout(layout)

    def __populate_connections_list(self):
        """
        Clear the list of connections that exist on the user's
        device and populate it with a new set of buttons.
        """

        # Clear it if it is populated already
        for i in reversed(range(self.__connections_list_layout.count())):
            self.__connections_list_layout.itemAt(i).widget().setParent(None)

        self.__connections_list_layout.addWidget(QLabel("Connections: "))
        no_connections = QLabel(
            'No connections found! Create a new one by pressing "New connection"'
        )

        try:
            connection_files = os.listdir(SAVED_CONNECTIONS_DIR)
        except FileNotFoundError:
            self.__connections_list_layout.addWidget(no_connections)
            return

        added_files = 0

        for connection_file in connection_files:
            if not connection_file.endswith(".json"):
                # Not a relevant file
                continue

            name = connection_file.removesuffix(".json")
            file = (SAVED_CONNECTIONS_DIR / connection_file).open("r")
            data = json.loads(file.read())
            btn = QPushButton(text=f"{name} ({data['model']} on {data['service']})")
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda: self.__select_connection(data))
            self.__connections_list_layout.addWidget(btn)
            added_files += 1

        if added_files == 0:
            self.__connections_list_layout.addWidget(no_connections)

    def __select_connection(self, data: dict[str, str]):
        """
        Called when a connection is selected.

        Args:
            data (dict[str, str]): A dictionary mapping a number of
                details about the connection to their respective
                values. Obrained from the JSON stored locally on
                the user's device.
        """

        self.parentWidget().setCentralWidget(VisualizationsScreen(data))

    def __show_new_connection_dialog(self):
        """
        Display the dialog box for creating a new connection.
        """

        dialog = NewConnectionDialog(callback=self.__populate_connections_list)
        dialog.exec()
