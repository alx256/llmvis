from IPython import get_ipython
from IPython.display import HTML
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
import sys
from pathlib import Path

from llmvis.visualization.visualization import Visualization

def start_visualization(visualization: Visualization):
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

    html = prelude + style + '<html><body>' + visualization.get_html() + '</body></html>'

    if 'zmqshell' in environment:
        # Jupyter Notebook
        return HTML(data = html)
    else:
        # Terminal
        app = QApplication(sys.argv)
        web = QWebEngineView()
        web.setWindowTitle('LLMVis')
        web.setHtml(html)
        web.show()
        sys.exit(app.exec())