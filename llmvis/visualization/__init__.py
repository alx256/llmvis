from IPython import get_ipython
from IPython.display import display, HTML, Javascript
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
import sys
import tempfile

from llmvis.visualization.visualization import Visualization, BACKGROUND_RGB_VALUE
from llmvis.visualization.linked_files import read_html, read_css

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
        is_jupyter = 'zmqshell' in environment

        # Embed CSS directly into the HTML
        style = read_css('css/style.css')
        prelude = read_html('html/prelude.html')
        rgb = f'rgb({BACKGROUND_RGB_VALUE}, {BACKGROUND_RGB_VALUE}, {BACKGROUND_RGB_VALUE})'

        html = '<html>'
        html += '<head>'
        html += prelude
        html += style
        html += '</head>'
        # html += font
        html += f'<body style="background-color: {rgb};">'
        html += visualization.get_html()
        if not is_jupyter:
            html += '<script>' + visualization.get_js() + '</script>'
        html += '</body>'
        html += '</html>'

        if is_jupyter:
            # Jupyter Notebook
            display(HTML(html))
            display(Javascript(visualization.get_js()))
        else:
            # Terminal
            web = QWebEngineView()
            web.setWindowTitle('LLMVis')
            web.setHtml(html)
            web.show()
            Visualizer.__app.exec()