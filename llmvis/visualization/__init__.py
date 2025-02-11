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

    def __init__(self, visualizations: list[Visualization]):
        """
        Create a new `Visualizer` that can be used to visualize a
        given list of `Visualization`s.

        Args:
            visualizations (list[Visualization]): the `Visualization`s
                that this `Visualizer` should show. Each `Visualization`
                will be able to be accessed from a row of tabs in the
                same order of this list where each tab can be used to
                show a different `Visualization`.
        """

        self.__visualizations = visualizations

    def show(self):
        """
        Show the `Visualization`s. These will either be rendered
        within a Jupyter notebook (if this is being run from
        a Jupyter notebook environment) or alternatively in the
        user's browser (if this is being run from any other
        environment, such as from a terminal).
        """

        environment = str(type(get_ipython()))
        is_jupyter = 'zmqshell' in environment

        if is_jupyter:
            # Jupyter Notebook
            display(HTML(self.get_html()))
            display(Javascript(self.get_js()))
        else:
            # Write the HTML data to a file first so we can just open
            # this using the browser
            llmvis_dir = pathlib.Path.home() / '.llmvis'

            if not os.path.isdir(llmvis_dir):
                os.mkdir(llmvis_dir)

            out = llmvis_dir / 'out.html'
            out.write_text(self.get_html() + '<script>' + self.get_js() + '</script>')

            webbrowser.open(f'file://{out}')
    
    def get_source(self) -> str:
        """
        Get the source code needed to show this `Visualizer`. This contains the
        HTML and JavaScript code that can be used to display all the
        `Visualization`s that this `Visualizer` has with tabs to switch between
        them.
        
        It is recommended that you use this when embedding a `Visualizer` in some
        web context. However, if you just need the HTML representation, use
        `get_html()` and if you just need the JavaScript code, use `get_js()`.

        Returns:
            A string containing HTML for showing the `Visualizer`, alongside
            necessary JavaScript functionality.
        """

        return self.get_html() + '<script>' + self.get_js() + '</script>'

    def get_html(self) -> str:
        """
        Get the HTML representation of this `Visualizer`. This can be
        displayed using any HTML renderer to show all the `Visualization`s
        that this `Visualizer` has with tabs to switch between them. Note
        that this is only the HTML representation without the JavaScript
        code that is needed to actually needed to provide functionality.
        You can get the JavaScript code by itself with `get_js()`, however
        if you are trying to embed this `Visualizer` in some web context,
        it is recommended that you just use `get_source()`.

        Returns:
            A string containing the HTML representation of this `Visualizer`.
        """

        # Embed CSS directly into the HTML
        style = read_css('css/style.css')
        prelude = read_html('html/prelude.html')
        rgb = f'rgb(23, 21, 33)'

        html = '<html>'
        html += '<head>'
        html += prelude
        html += style
        html += '</head>'
        html += f'<body style="background-color: {rgb}; margin: 0px;">'
        html += '<div id="llmvis-tabs-container">'
        html += '<div class="llmvis-tabs">'
        for i, visualization in enumerate(self.__visualizations):
            html += f'<button class="llmvis-tab" onclick="openVisualization({i})">'
            html += '<div class="llmvis-text">'
            html += visualization.get_name()
            html += '</div>'
            html += '</button>'
        html += '</div>'
        html += '</div>'
        for visualization in self.__visualizations:
            html += '<div class="llmvis-visualization-content">'
            html += visualization.get_html()
            html += '</div>'
        html += '</body>'
        html += '</html>'

        return html

    def get_js(self) -> str:
        """
        Get a JavaScript script containing everything that is needed
        in one place. This includes the required JS scripts as well as
        scripts that independent visualizations may (or may not) provide.
        
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
        added = set()

        for visualization in self.__visualizations:
            # Load the files that these visualizations
            # are dependent on, but only once to prevent
            # conflicts.
            for dependency in visualization.get_dependencies():
                if dependency not in added:
                    script += relative_file_read(dependency)
                    added.add(dependency)

            script += visualization.get_js()

        return script