import abc
import math
import re
from numpy.typing import ArrayLike
from typing import Optional
import uuid

from llmvis.core import js_tools
from llmvis.core.js_tools import escape_all, list_as_js


class Unit:
    """
    An individual unit that will be shown by the `TextHeatmap`
    visualization. This is commonly a full word, but smaller
    components of words or large clusters of words can be
    units too.
    """

    def __init__(self, text: str, weight: float, details: list[tuple[str, any]]):
        """
        Create a new `Unit`.

        Args:
            text (str): The text contents of the unit. This is
                what will be shown in the heatmap.
            weight (float): A numerical weight representing
                the color of the unit when it is displayed on
                the heatmap. Positive values are shown as red
                with higher values being stronger shades of red
                while negative values are shown as blue with
                lower values being stronger shades of blue.
            details (list[tuple[str, any]]): A list containing
                all the details of the unit. This represents
                additional information that should be shown when
                the user hovers or clicks on the unit. Each element
                should be a tuple where the first element is the
                name of the detail and the second is the associated
                data.
        """
        self.text = text
        self.weight = weight
        self.details = details

    def get_js(self):
        """
        Get a JavaScript representation of this `Unit`.

        Returns:
            A JavaScript representation of this `Unit` as an object
            containing the attributes of the `Unit`. Note that
            details is encoded as 2D list instead of a list of
            tuples which is its initial representation due to
            JavaScript not supporting tuples.
        """

        js = "{"
        js += f"text:'{escape_all(self.text)}',"
        js += f"weight:'{escape_all(str(self.weight))}',"
        js += "details: ["

        for i, detail in enumerate(self.details):
            js += f"['{escape_all(str(detail[0]))}', '{escape_all(str(detail[1]))}']"

            if i < len(self.details) - 1:
                js += ","

        js += "]}"

        return js


class Point:
    """
    A 2-dimensional point. Useful for a wide variety of graphs
    that make use of both an x and a y axis.
    """

    def __init__(
        self,
        x: float,
        y: float,
        detail: Optional[str] = None,
        color: str = "rgb(180, 157, 46)",
    ):
        """
        Create a new `Point`.

        Args:
            x (int): The x position of this point.
            y (int): The y position of this point.
            detail (Optional[str]): The details that are associated with
                this point (can be none), commonly used to include
                additional information for tooltips.
            color (str): The color that this point should be (if supported
                by the visualization). Default is a yellow color.
        """

        self.x = x
        self.y = y
        self.detail = detail
        self.color = color

    def get_js(self):
        """
        Get a JavaScript representation of this `Point`.

        Returns:
            A string containing a JavaScript representation of
            this `Point` data.
        """

        js = "{"
        js += f"x:{self.x},"
        js += f"y:{self.y}"
        if self.detail is not None:
            js += f',detail:"{escape_all(self.detail)}"'
        if self.color is not None:
            js += f',color:"{escape_all(self.color)}"'
        js += "}"

        return js


class Visualization(abc.ABC):
    WIDTH = 600
    HEIGHT = 300

    """
    Base class for a visualization. Used to define the HTML
    representation of a specific `Visualization` so it can
    be rendered.
    """

    def __init__(self):
        """
        Create a new `Visualization`.
        """

        self.__uuid__ = None
        self.__comments__ = []
        self.__name__ = "Unnamed Visualization"

    def get_name(self) -> str:
        """
        Get the name of this `Visualization`. Useful for cases
        where a nicely formatted, descriptive name of this
        `Visualization` is needed.

        Returns:
            A string containing the name of this `Visualization`.
        """
        return self.__name__

    @abc.abstractmethod
    def get_html(self) -> str:
        """
        Get the HTML representation of this `Visualization`.

        Returns:
            A string containing the HTML representation of this
                `Visualization`.
        """

        pass

    @abc.abstractmethod
    def get_js(self) -> str:
        """
        Get the JavaScript representation of this `Visualization`.
        Used to provide the code for drawing to the canvas provided
        by `get_html` (if this visualization uses a canvas).

        Returns:
            A string containing the JavaScript representation of
                this `Visualization`.
        """

        pass

    def get_dependencies(self) -> list[str]:
        """
        Use this to load core JavaScript components of the visualization
        but only once since the contents of each dependency has the
        potential to cause conflicts if it is added multiple times to
        the final file.

        Returns:
        A list of strings containing the relative paths for
        JavaScript files containing core functions that are required
        for this visualization.
        """
        return []

    def set_comments(self, *comments: str):
        """
        Set the comments for this `Visualization`. Comments are shown below
        the visualization and should contain some information about what
        the visualization is showing.

        Args:
            comments (str): Arg list of comments that should be shown under
                the visualization. Each will use a new line. Since this is
                embedded directly into the HTML, any HTML is valid here.
        """

        self.__comments__ = comments

    def set_name(self, name: str):
        """
        Set the name of this `Visualization` to a custom one. This will
        update the name shown when this is visualized using a `Visualizer`,
        so it is helpful for setting a more descriptive name for this
        `Visualization` than what might be used by default.
        """

        self.__name__ = name

    def call_function(self, func: str, *args: list[any]) -> str:
        """
        Get the JavaScript code to load fonts and then call a
        provided function immediately afterwards.

        Args:
            func (str): The name of the function that should be
                called.
            args (list[any]): The arguments (if any) that should
                be passed into the function.

        Returns:
            A string containing the JavaScript code that loads the
                necessary fonts and calls the function when this is
                done.
        """
        js = "loadFonts().then(function(){"
        js += f"{func}("

        for i, arg in enumerate(args):
            js += str(arg)

            if i < len(args) - 1:
                js += ","

        js += ")"
        js += "});"
        return js

    def get_uuid(self):
        """
        Get a UUID. Useful for canvases where each one should have
        its own unique ID to prevent conflicts. Note that a UUID
        will only be created once for each `Visualization` so if
        multiple elements in the HTML need to have different IDs,
        you should use variations of this one since a new UUID will
        not be given each time this method is called.

        Returns:
        A `UUID` object containing the `UUID` that has been generated
        for this `Visualization`.
        """

        if self.__uuid__ is None:
            self.__uuid__ = uuid.uuid4()

        return self.__uuid__

    def get_comments_html(self) -> str:
        """
        Get the HTML for the comments of this `Visualization`.
        Comments are set using `set_comments` and contain
        details about this `Visualization` that should be
        displayed under the `Visualization` itself.

        Returns:
        A string containing the HTML representation of the comments
        for this visualization.
        """

        html = ""

        for comment in self.__comments__:
            html += (
                '<div class="llmvis-text" style="padding:20px;height:2%;">'
                + comment
                + "</div>"
            )

        return html


class HeatmapColorScheme:
    """
    The color scheme that a heatmap should use. This will determine
    how different components are colored to indicate "hot" vs "cold"
    values.

    Values:
    - `BLUE_RED`: Colder values use bluer colors, hotter values use
        redder colors. Good for classic heatmap behaviour using
        colors that humans typically associate with heat.
    - `RED_BLUE`: Inverse of `BLUE_RED`. Hotter values use bluer colors,
        colder values use redder colors.
    - `GREEN_RED`: Colder values use greener colors, hotter values use
        redder colors. Good for heatmap data, especially data that can be
        both hot and cold, where you want to indicate that hotter
        values are "bad" while colder values are "good".
    - `RED_GREEN`: Colder values use redder colors, hotter values use
        greener colors. Good for heatmap data, especially data that can be
        both hot and cold, where you want to indicate that cold
        values are "good" while hot values are "bad".
    - `GREEN_YELLOW_RED`: Colder values use greener colors, neutral values
        use yellow colors and hotter values use redder colors. Good for
        data that has more of an emphasis on neutrality.
    - `RED_YELLOW_GREEN`: Colder values use redder colors, neutral values
        use yellow colors and hotter values use greener colors. Good for
        data that has more of an emphasis on neutrality.
    """

    BLUE_RED = 0
    RED_BLUE = 1
    GREEN_RED = 2
    RED_GREEN = 3
    GREEN_YELLOW_RED = 4
    RED_YELLOW_GREEN = 5


class TextHeatmap(Visualization):
    """
    A heatmap for a chunk of text where each individual unit
    of the text (e.g. a word in the text) is colored depending
    on a corresponding weight.
    """

    def __init__(
        self,
        units: list[Unit],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        color_scheme: HeatmapColorScheme = HeatmapColorScheme.BLUE_RED,
        relative_coloring: bool = False,
    ):
        """
        Create a new `TextHeatmap` for a provided list of `Unit`s.

        Args:
            units (list[Unit]): A list of `Unit`s (such as words) that
                make up the chunk of text that should be
                visualized.
            min_value (Optional[float]): An optional minimum value that
                will be used for the colouring or `None` to disable and
                automatically use the maximum weight of the units.
                Default is `None`.
            max_value (Optional[float]): An optional maximum value that
                will be used for the colouring or `None` to disable and
                automatically use the maximum weight of the units.
                Default is `None`.
            color_scheme (HeatmapColorScheme): The `HeatmapColorScheme`
                that should be used for coloring units. Default is
                `BLUE_RED`, using blue for colder values and red for
                hotter values.
            relative_coloring (bool): Set this to `True` to use "relative
                coloring", where the smallest values use the coldest colors
                and the largest values use the hottest colors. `False` only
                uses colder colors for negative values and hotter colors for
                positive ones. Default is `False`.
        """

        super().__init__()
        self.__units = units
        self.__name__ = "Text Heatmap"

        max_weight = None
        min_weight = None

        if min_value is not None and max_value is not None:
            # Both a maximum weight and a minimum weight has
            # been defined, so no need to automatically find
            # this.
            max_weight = max_value
            min_weight = min_value
        else:
            for unit in units:
                if max_weight is None or max_weight < unit.weight:
                    max_weight = unit.weight

                if min_weight is None or min_weight > unit.weight:
                    min_weight = unit.weight

        self.__max_weight = max_weight
        self.__min_weight = min_weight
        self.__color_scheme__ = color_scheme
        self.__relative_coloring__ = relative_coloring

    def get_html(self) -> str:
        html = f'<canvas id="{self.get_uuid()}" width="{self.WIDTH}" height="{self.HEIGHT}">'
        html += "</canvas>"

        return html

    def get_js(self):
        units_str = "["

        for i, unit in enumerate(self.__units):
            units_str += unit.get_js()

            if i < len(self.__units) - 1:
                units_str += ","

        units_str += "]"

        js = self.call_function(
            "drawHeatmap",
            f'"{self.get_uuid()}"',
            units_str,
            self.__min_weight,
            self.__max_weight,
            self.__color_scheme__,
            "true" if self.__relative_coloring__ else "false",
        )
        return js

    def get_dependencies(self):
        return ["visualization/js/heatmap.js"]


class TableHeatmap(Visualization):
    """
    A heatmap in the form of a table containing a number of rows
    and each row is colored depending on a corresponding weight.
    """

    GREY_VALUE = 61

    def __init__(
        self,
        headers: list[str],
        contents: list[list[str]],
        weights: list[float] = [],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Create a new `TableHeatmap` for some provided content and weights.

        Args:
            headers (list[str]): A list of strings where each string corresponds to the
                header for the corresponding column. Length must be equal to the length
                of each row (i.e. the number of columns).
            contents (list[list[str]]): A list where each element represents a row,
                containing another list with the content that should be contained in
                the table.
            weights (list[float]): A list where each float corresponds to the weight for
                the corresponding row. If no weight is specified for a row, it will be
                colored a default grey color. Providing an empty list or not giving a
                value will give a table without any coloring.
            min_value (Optional[float]): The custom minimum value of all the weights.
                Leave as `None` to automatically detect this.
            max_value (Optional[float]): The custom maximum value of all the weights.
                Leave as `None` to automatically detect this.

        Raises:
            `ValueError` if the lengths of the provided arguments are incorrect or if the table is empty.
        """

        if len(contents) == 0:
            raise ValueError("Table cannot be empty!")

        if len(headers) != len(contents[0]):
            raise ValueError("headers must be equal to the number of columns!")

        super().__init__()

        self.__headers = headers
        self.__contents = contents
        self.__weights = weights
        self.__name__ = "Table Heatmap"

        # Explanation for doing this explained in TextHeatmap above
        if len(weights) > 0:
            self.__max_weight = max_value or max(
                weights, key=lambda x: x if x is not None else -math.inf
            )
            self.__min_weight = min_value or min(
                weights, key=lambda x: x if x is not None else math.inf
            )

    def get_html(self):
        html = "<table>"
        html += "<colgroup>"
        html += '<col span="1" style="width: 25%;">'
        html += '<col span="1" style="width: 75%;">'
        html += "</colgroup>"

        html += "<tbody>"
        html += "<tr>"
        for cell in self.__headers:
            html += "<th>"
            html += '<div class="llmvis-text">'
            html += cell
            html += "</div>"
            html += "</th>"
        html += "</tr>"
        for i, content in enumerate(self.__contents):
            html += "<tr>"
            for entry in content:
                bg = self.__get_background_color(
                    None if i >= len(self.__weights) else self.__weights[i],
                    None if i >= len(self.__weights) else self.__min_weight,
                    None if i >= len(self.__weights) else self.__max_weight,
                )
                html += f'<td style="background-color: {bg};">'
                html += '<div class="llmvis-text">'
                html += str(entry)
                html += "</div>"
                html += "</td>"
            html += "</tr>"
        html += "</tbody>"
        html += "</table>"

        return html

    def get_js(self):
        return ""

    def __get_background_color(
        self, weight: Optional[float], min_weight: float, max_weight: float
    ) -> str:
        """
        Calculate the necessary background color for a row
        based on a provided weight.

        Args:
            weight (float): The weight that should be used for
                calculating the background color.
            min_weight (float): The minimum of all the weights.
            max_weight (float): The maximum of all the weights.

        Returns:
            A string representing the CSS `background-color`
            property that should be used for coloring the row
            with this weight.
        """
        palette = [[48, 147, 38], [180, 157, 46], [176, 46, 52]]

        if weight is None:
            return f"rgb({self.GREY_VALUE}, {self.GREY_VALUE}, {self.GREY_VALUE})"

        normalized_weight = (
            0
            if max_weight - min_weight == 0
            else (weight - min_weight) / (max_weight - min_weight)
        )
        index = (
            normalized_weight * (len(palette) - 1)
            if max_weight != min_weight
            else (len(palette) - 1) / 2
        )

        if index % 1 == 0:
            # Nicely falls onto a usable index
            return f"rgb({palette[int(index)][0]}, {palette[int(index)][1]}, {palette[int(index)][2]})"

        if index > len(palette) - 1:
            return f"rgb({palette[-1][0]}, {palette[-1][1]}, {palette[-1][2]})"

        rgb = [0, 0, 0]
        index_lower = math.floor(index)
        index_upper = index_lower + 1
        diff = index - index_lower
        rgb_lower = palette[index_lower]
        rgb_upper = palette[index_upper]

        rgb[0] = (rgb_upper[0] - rgb_lower[0]) * diff + rgb_lower[0]
        rgb[1] = (rgb_upper[1] - rgb_lower[1]) * diff + rgb_lower[1]
        rgb[2] = (rgb_upper[2] - rgb_lower[2]) * diff + rgb_lower[2]

        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


class TagCloud(Visualization):
    """
    A "Tag Cloud" `Visualization` (also known as a word cloud). Used to
    visualize some `Unit`s by showing ones with higher weights as bigger
    and ones with lower weights as smaller.
    """

    def __init__(self, units: list[Unit]):
        """
        Create a new `TagCloud` `Visualization` from a list of `Unit`s.

        Args:
            units (list[Unit]): The list of units which will be visualized
                by this `TagCloud`.
        """
        super().__init__()
        self.__units = units
        self.__name__ = "Tag Cloud"

    def get_html(self):
        html = f'<canvas id="{self.get_uuid()}" width="{self.WIDTH}" height="{self.HEIGHT}">'
        html += "</canvas>"

        return html

    def get_js(self):
        units_str = "["

        for i, unit in enumerate(self.__units):
            units_str += unit.get_js()
            if i < len(self.__units) - 1:
                units_str += ","

        units_str += "]"
        js = self.call_function("drawTagCloud", f'"{self.get_uuid()}"', units_str)

        return js

    def get_dependencies(self):
        return ["visualization/js/tag_cloud.js"]


class ScatterPlot(Visualization):
    """
    A Scatter Plot for some data. This will visualize
    an array-like object of 2D points in 2D space.
    """

    def __init__(
        self, plots: list[Point], x_axis_label: str = "", y_axis_label: str = ""
    ):
        """
        Create a new `ScatterPlot` `Visualization` from a
        2D array of plots.

        Args:
            plots (list[Point]): A list of `Point` objects
                representing the different 2D points that
                this scatter plot should show.
            x_axis_label (str): The label that should be
                shown on the x-axis. Default is an empty
                label.
            y_axis_label (str): The label that should be
                shown on the y-axis. Default is an empty
                label.
        """

        super().__init__()
        self.__plots = plots
        self.__name__ = "Scatter Plot"
        self.__x_axis_label__ = x_axis_label
        self.__y_axis_label__ = y_axis_label

    def get_html(self):
        html = f'<canvas id="{self.get_uuid()}" width="700" height="700">'
        html += "</canvas>"

        return html

    def get_js(self):
        return self.call_function(
            "drawScatterPlot",
            f'"{self.get_uuid()}"',
            js_tools.list_as_js(self.__plots, do_conversion=True),
            f'"{self.__x_axis_label__}"',
            f'"{self.__y_axis_label__}"',
        )

    def get_dependencies(self):
        return ["visualization/js/scatter_plot.js"]


class BarChart(Visualization):
    """
    A Bar Chart for some data. Given a list of categorical
    values and their associated numerical (real or integer)
    values, display a coloured bar chart for each value.
    """

    def __init__(
        self, values: list[list[any]], x_axis_label: str = "", y_axis_label: str = ""
    ):
        """
        Create a new `BarChart` `Visualization` from a list
        of values.

        Args:
            values (list[list[any]]): A list of values that
                the bar chart will visualize. Each element of the
                list should be another list where the first element
                is the categorical value that will be shown on the
                x-axis and the second element is the numerical (real
                or integer) value that will be shown on the y-axis.
            x_axis_label (str): The label that should be displayed
                under the x-axis in order to describe it. Can be empty
                to show no label. Default is empty string.
            y_axis_label (str): The label that should be displayed
                next to the y-axis in order to describe it. Can be empty
                to show no label. Default is empty string.
        """
        super().__init__()
        self.__values = values
        self.__name__ = "Bar Chart"
        self.__x_axis_label__ = x_axis_label
        self.__y_axis_label__ = y_axis_label

    def get_html(self) -> str:
        html = f'<canvas id="{self.get_uuid()}" width="500" height="500">'
        html += "</canvas>"

        return html

    def get_js(self) -> str:
        return self.call_function(
            "drawBarChart",
            f'"{self.get_uuid()}"',
            self.__get_js_values(),
            f'"{self.__x_axis_label__}"',
            f'"{self.__y_axis_label__}"',
        )

    def get_dependencies(self):
        return ["visualization/js/bar_chart.js"]

    def __get_js_values(self) -> str:
        """
        Convert the `values` list to a string containing the list in
        JavaScript syntax so that it can be embedded into the JavaScript
        code.

        Returns:
            A string containing the `values` list as a JavaScript list.
        """
        js = "["

        for i, (name, value) in enumerate(self.__values):
            js += f'["{name}",{value}]'

            if i < len(self.__values) - 1:
                js += ","

        js += "]"
        return js


class LineChart(Visualization):
    """
    A line chart visualization for some data. Display a number of points
    that are connected by lines.
    """

    def __init__(
        self, values: list[Point], x_axis_label: str = "", y_axis_label: str = ""
    ):
        """
        Create a new `LineChart` `Visualization` for a list of values.

        Args:
            values (list[list[any]]): A list of values that the line
                chart will visualize. Each element of the list should
                be another list where the first element is the categorical
                or numerical value that will be shown on the x-axis and
                the second element is the numerical value (real or integer)
                that will be shown on the y-axis. There must be a minimum
                of two values provided.
            x_axis_label (str): The label that should be shown on the
                x-axis. Default is an empty label.
            y_axis_label (str): The label that should be shown on the
                y-axis. Default is an empty label.
        """

        assert len(values) >= 2
        super().__init__()
        self.__values = values
        self.__name__ = "Line Chart"
        self.__x_axis_label__ = x_axis_label
        self.__y_axis_label__ = y_axis_label

    def get_html(self) -> str:
        html = f'<canvas id="{self.get_uuid()}" width="500" height="500">'
        html += "</canvas>"

        return html

    def get_js(self) -> str:
        return self.call_function(
            "drawLineChart",
            f'"{self.get_uuid()}"',
            list_as_js(self.__values, do_conversion=True),
            f'"{self.__x_axis_label__}"',
            f'"{self.__y_axis_label__}"',
        )

    def get_dependencies(self):
        return ["visualization/js/line_chart.js"]


class RadarChart(Visualization):
    """
    A radar chart visualization for some data. Display a polygon
    where each point represents some numerical data.
    """

    def __init__(self, values: list[list[any]]):
        """
        Create a new `RadarChart` `Visualization`.

        Args:
            values (list[list[any]]): The data that this `RadarChart`
                will show. Each element should be a 2-dimensional list
                where the first element is the categorical name of the
                data point and the second element is the numerical
                value.
        """

        super().__init__()
        self.__values__ = values
        self.__name__ = "Radar Chart"

    def get_html(self) -> str:
        html = f'<canvas id="{self.get_uuid()}" width="500" height="500">'
        html += "</canvas>"
        return html

    def get_js(self) -> str:
        return self.call_function(
            "drawRadarChart", f'"{self.get_uuid()}"', self.__values__
        )

    def get_dependencies(self):
        return ["visualization/js/radar_chart.js"]
