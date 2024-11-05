from IPython import get_ipython
from IPython.display import HTML
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
import sys
from pathlib import Path

from llmvis.visualization.visualization import Visualization, BACKGROUND_RGB_VALUE

class Visualizer():
    """
    Used to visualize some data either in a Jupyter notebook
    environment or as a standalone window.
    """

    __app = QApplication(sys.argv)

    def start_visualization(self, visualization: Visualization):
        """
        Open a visualization. This will either be rendered
        within a Jupyter notebook (if this is being run from
        a Jupyter notebook environment) or alternatively as a
        standalone window (if this is being run from any other
        environment, such as from a terminal).

        Args:
            visualization (Visualization): The visualization that
                should be started
        """

        environment = str(type(get_ipython()))

        # Calculate the relative path for the visualization/ directory
        root_path = Path(__file__).parent
        style_path = root_path / 'css/style.css'
        prelude_path = root_path / 'html/prelude.html'

        # Embed CSS directly into the HTML
        style = '<style>' + style_path.open().read() + '</style>' + '\n'
        prelude = prelude_path.open().read() + '\n'

        rgb = f'rgb({BACKGROUND_RGB_VALUE}, {BACKGROUND_RGB_VALUE}, {BACKGROUND_RGB_VALUE})'
        html = prelude + style + f'<html><body style="background-color: {rgb};">' + visualization.get_html() + '</body></html>'

        if 'zmqshell' in environment:
            # Jupyter Notebook
            return HTML(data = html)
        else:
            # Terminal
            web = QWebEngineView()
            web.setWindowTitle('LLMVis')
            web.setHtml(html)
            web.show()
            Visualizer.__app.exec()