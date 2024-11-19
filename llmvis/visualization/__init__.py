from IPython import get_ipython
from IPython.display import display, HTML, Javascript
import os
import pathlib
import webbrowser

from llmvis.visualization.visualization import Visualization
from llmvis.visualization.linked_files import read_html, read_css, relative_file_read

class Visualizer():
    """
    Used to visualize some data either in a Jupyter notebook
    environment or as a standalone window.
    """

    def start_visualizations(self, visualizations: list[Visualization]):
        """
        Open some `Visualization`s. These will either be rendered
        within a Jupyter notebook (if this is being run from
        a Jupyter notebook environment) or alternatively in the
        user's browser (if this is being run from any other
        environment, such as from a terminal).

        Args:
            visualizations (list[Visualization): The `Visualization`s
                that should be started
        """

        environment = str(type(get_ipython()))
        is_jupyter = 'zmqshell' in environment

        # Embed CSS directly into the HTML
        style = read_css('css/style.css')
        prelude = read_html('html/prelude.html')
        rgb = f'rgb(23, 21, 33)'

        js = self.__generate_js_script(visualizations)

        html = '<html>'
        html += '<head>'
        html += prelude
        html += style
        html += '</head>'
        html += f'<body style="background-color: {rgb}; margin: 0px;">'
        html += '<div id="llmvis-tabs-container">'
        html += '<div class="llmvis-tabs">'
        for i, visualization in enumerate(visualizations):
            html += f'<button class="llmvis-tab" onclick="openVisualization({i})">'
            html += '<div class="llmvis-text">'
            html += visualization.get_name()
            html += '</div>'
            html += '</button>'
        html += '</div>'
        html += '</div>'
        for visualization in visualizations:
            html += '<div class="llmvis-visualization-content">'
            html += visualization.get_html()
            html += '</div>'
        if not is_jupyter:
            html += '<script>' + js + '</script>'
        html += '</body>'
        html += '</html>'

        if is_jupyter:
            # Jupyter Notebook
            display(HTML(html))
            display(Javascript(js))
        else:
            # Write the HTML data to a file first so we can just open
            # this using the browser
            llmvis_dir = pathlib.Path.home() / '.llmvis'

            if not os.path.isdir(llmvis_dir):
                os.mkdir(llmvis_dir)

            out = llmvis_dir / 'out.html'
            out.write_text(html)

            webbrowser.open(f'file://{out}')
    
    def __generate_js_script(self, visualizations: list[Visualization]) -> str:
        """
        Generate a JavaScript script containing everything that is needed
        in one place. This includes the required JS scripts as well as
        scripts that independent visualizations may (or may not) provide.

        Args:
            visualizations (list[Visualization]): The `Visualization`s that are
                being started. Needed so any JS code for each visualization can
                be included.
        
        Returns:
            A string representation of the JS script that can be either directly
            embedded into HTML between `<script>` tags or executed independently
            through IPython's `Javascript` object.
        """
        script = ''

        # Add required scripts
        script += relative_file_read('js/core.js')
        script += relative_file_read('js/tabs.js')

        # Add any scripts from visualizations
        for visualization in visualizations:
            script += visualization.get_js()

        return script