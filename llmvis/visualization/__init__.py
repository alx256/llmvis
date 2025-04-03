from __future__ import annotations
from IPython import get_ipython
from IPython.display import display, HTML, Javascript
import os
import pathlib
import webbrowser

from llmvis.visualization.visualization import Visualization
from llmvis.visualization.linked_files import read_html, read_css, relative_file_read


class Visualizer:
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

        self.__visualizations = {}

        # Visualizations dictionary should map a visualization's name
        # to a list of visualizations that should be shown side by side.
        for v in visualizations:
            self.__visualizations[v.get_name()] = [v]

    def show(self):
        """
        Show the `Visualization`s. These will either be rendered
        within a Jupyter notebook (if this is being run from
        a Jupyter notebook environment) or alternatively in the
        user's browser (if this is being run from any other
        environment, such as from a terminal).
        """

        environment = str(type(get_ipython()))
        is_jupyter = "zmqshell" in environment
        full_html = self.get_html() + "<script>" + self.get_js() + "</script>"

        if is_jupyter:
            # Jupyter Notebook
            display(HTML(full_html))
        else:
            # Write the HTML data to a file first so we can just open
            # this using the browser
            llmvis_dir = pathlib.Path.home() / ".llmvis"

            if not os.path.isdir(llmvis_dir):
                os.mkdir(llmvis_dir)

            out = llmvis_dir / "out.html"
            out.write_text(
                full_html,
                encoding="utf-8",
            )

            webbrowser.open(f"file://{out}")

    def merge(self, other: Visualizer):
        """
        Merge this `Visualizer` with another, showing the `Visualization`s
        that the two `Visualizer`s have in common side by side and adding
        any new `Visualization`s that the other `Visualizer` has.

        Args:
            other (Visualizer): The other `Visualizer` that this `Visualizer`
                should be merged with.
        """

        remaining_keys = set(other.__visualizations.keys())

        for key in other.__visualizations.keys():
            if key in self.__visualizations:
                self.__visualizations[key] += other.__visualizations[key]
                remaining_keys.remove(key)

        for remaining in remaining_keys:
            self.__visualizations[remaining] = other.__visualizations[remaining]

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

        return self.get_html() + "<script>" + self.get_js() + "</script>"

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
        style = read_css("visualization/css/style.css")
        prelude = read_html("visualization/html/prelude.html")
        rgb = f"rgb(23, 21, 33)"

        html = "<!DOCTYPE html>"
        html += '<html style="height:100%;">'
        html += "<head>"
        html += "<title>"
        html += "LLMVis"
        html += "</title>"
        html += prelude
        html += style
        html += "</head>"
        html += f'<body style="height:100%;background-color: {rgb}; margin: 0px;display:flex;flex-direction:column;">'
        html += '<div id="llmvis-tabs-container" style="flex:1;">'
        html += '<div class="llmvis-tabs">'
        for i, name in enumerate(self.__visualizations.keys()):
            html += f'<button class="llmvis-tab" onclick="openVisualization({i})">'
            html += '<div class="llmvis-text">'
            html += name
            html += "</div>"
            html += "</button>"
        html += "</div>"
        html += "</div>"
        for name in self.__visualizations.keys():
            html += '<div class="llmvis-visualization-content" style="flex:9;">'
            html += '<div class="llmvis-flex-container" style="display:flex;flex-direction:row;overflow:hidden;">'
            for v in self.__visualizations[name]:
                html += '<div class="llmvis-flex-child" style="flex:1;overflow:auto;max-width:100%;">'
                html += v.get_html()
                html += v.get_comments_html()
                html += "</div>"
            html += "</div>"
            html += "</div>"
        html += "</body>"
        html += "</html>"

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
        script = ""

        # Add required scripts
        script += relative_file_read("visualization/js/core.js")
        script += relative_file_read("visualization/js/tabs.js")

        # Add any scripts from visualizations
        added = set()

        for name in self.__visualizations.keys():
            # Load the files that these visualizations
            # are dependent on, but only once to prevent
            # conflicts.
            for dependency in self.__visualizations[name][0].get_dependencies():
                if dependency not in added:
                    script += relative_file_read(dependency)
                    added.add(dependency)

            for subvis in self.__visualizations[name]:
                script += subvis.get_js()

        return script
