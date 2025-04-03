from typing import Optional

import html as htmlmanip

from llmvis.core import js_tools
from llmvis.visualization import Visualization
from llmvis.visualization.visualization import LineChart, RadarChart, Point
from llmvis.visualization.linked_files import relative_file_read
from llmvis.core.js_tools import escape_all, list_as_js


class WordSpecificLineChart(LineChart):
    """
    Modified `LineChart` that also includes a text box where the user
    can enter in the specific line chart that they wish to see. For example,
    if we have a different line chart for the frequency of each word as some
    parameter changes, this visualization allows the user to conveniently
    switch between the line charts for all the different words.
    """

    def __init__(
        self,
        word_values: dict[str, list[Point]],
        t_values: list[int],
        x_axis_label: str = "",
        y_axis_label: str = "",
    ):
        """
        Create a new `WordSpecificLineChart` `Visualization`.

        Args:
            word_values (dict[str, list[Point]]): A dictionary mapping each word
                to a `Point` that will be shown on the line chart. See `LineChart`
                and `Point` for information on how this data should be formatted.
            t_values (list[int]): A list of integers for the complete temperature
                values that were used for sampling. This is required so that if
                in a given word it is missing a frequency for a given temperature,
                this can be treated as zero in the visualization and thus the line
                chart x axis contains the same values for all words.
            x_axis_label (str): The label that should be shown on the x-axis.
                Default is an empty label.
            y_axis_label (str): The label that should be shown on the y-axis.
                Default is an empty label.
        """

        self.__word = list(word_values.keys())[0]
        self.__word_values = word_values
        self.__t_values = t_values

        super().__init__(
            self.__filled_list(self.__word),
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )
        self.__textbox_id = str(self.get_uuid()) + "0"
        self.__name__ = "Word-Specific Line Chart"

    def get_html(self):
        html = (
            '<input type="text" class="llmvis-textbox" placeholder="Enter a word..." '
        )
        html += f'id="{self.__textbox_id}" value="{self.__word}">'
        html += "<br>"
        return html + super().get_html()

    def get_js(self):
        js = self.call_function(
            "connectFieldToLineChart",
            f'"{self.get_uuid()}"',
            f'"{self.__textbox_id}"',
            self.__values_as_object(),
            f'"{self.__x_axis_label__}"',
            f'"{self.__y_axis_label__}"',
        )
        return js + super().get_js()

    def get_dependencies(self):
        return ["js/word_specific_line_chart.js"] + super().get_dependencies()

    def __values_as_object(self):
        """
        Get the values of the `word_values` dictionary as a string
        representing a JavaScript object so that it can be inserted
        into the JavaScript representation of this visualization.

        Returns:
            A string containing the `word_values` dictionary as a
            JavaScript object.
        """

        # Format:
        # {
        # "word_1" : <filled list of line chart values for word_1>
        # "word_2" : <filled list of line chart values for word_2>
        # ...
        # "word_n" : <filled list of line chart values for word_n>
        # }

        obj = "{"

        for i, key in enumerate(self.__word_values.keys()):
            obj += (
                '"'
                + key
                + '"'
                + ":"
                + list_as_js(self.__filled_list(key), do_conversion=True)
            )

            if i < len(self.__word_values.keys()) - 1:
                obj += ","

        obj += "}"

        return obj

    def __filled_list(self, word: str) -> str:
        """
        Get a string containing raw line chart data for a given
        `word`, with any missing values filled as 0. Useful for
        giving a consistent line chart x-axis between different
        charts.

        Args:
            word (str): The word that should have its raw line
                chart data filled.

        Returns:
            A string containing the JavaScript representation of
            the filled list.
        """

        values = self.__word_values[word]

        # Already contains everything, we can just use it as-is
        if len(values) == len(self.__t_values):
            return values

        i = 0
        filled = []

        for t in self.__t_values:
            if i < len(values) and values[i].x == t:
                filled.append(values[i])
                i += 1
            else:
                filled.append(Point(t, 0.0))

        return filled


class AIClassifier(Visualization):
    """
    `Visualization` that, given some 1D data that has been assigned to classes,
    displays each of the classes on the y-axis, the data points on the x-axis,
    and shows a rectangle to visualize which data points belong to which class.
    """

    def __init__(
        self,
        items: Optional[list[any]],
        t_values: list[float],
        x_axis_label: str = "",
        y_axis_label: str = "",
    ):
        """
        Create a new `AIClassifier` `Visualization`.

        Args:
            items (Optional[list[any]]): The classified data. Each element of
                this list should contain another list where the first element
                is a string containing the class name and the second element is
                a list of all the data points belonging to that class. Can also
                be `None` to indicate that there was a problem with the
                classification and this `Visualization` should show an error
                instead.
            t_values (list[float]): A list of all the data points. Note that
                all data points that have been classified and are therefore
                present in `items` should be present in this list, however not
                all the points in this list need to necessarily be present in
                `items`.
            x_axis_label (str): The label that should be used for the x-axis of
                this visualization. Default is an empty label.
            y_axis_label (str): The label that should be used for the y-axis of
                this visualization. Default is an empty label.
        """

        super().__init__()
        self.__items = items
        self.__t_values = t_values
        self.__name__ = "AI Classifier"
        self.__x_axis_label__ = x_axis_label
        self.__y_axis_label__ = y_axis_label

    def get_html(self) -> str:
        if self.__items is None:
            return "<p>Failed to classify data</p>"

        html = f'<canvas id="{self.get_uuid()}" width="1280" height="500"'
        html += "</canvas>"
        return html

    def get_js(self) -> str:
        if self.__items is None:
            return ""

        return self.call_function(
            "drawAiClassifier",
            f'"{self.get_uuid()}"',
            self.__items,
            self.__t_values,
            f'"{self.__x_axis_label__}"',
            f'"{self.__y_axis_label__}"',
        )

    def get_dependencies(self):
        return ["js/ai_classifier.js"]


class Token:
    """
    Representation of a token that a model might return,
    containing the `text` of the token as well as the
    `prob` (probability) that this token should be
    next in the sequence.
    """

    def __init__(self, text: str, prob: float):
        """
        Create a new `Token`.

        Args:
            text (str): The text contents of this token.
            prob (float): The log probability that this
                token will be next in the sequence.
        """

        self.text = text
        self.prob = prob

        # Process text
        self.text = self.text.replace("\n", "<newline>")
        self.text = self.text.replace("\t", "<tab>")
        self.text = self.text.replace("\r", "<return>")
        self.text = self.text.replace("\f", "<formfeed>")

    def get_js(self):
        """
        Get a JS representation of this token.

        Returns:
            A string containing JavaScript code for an
            JavaScript object representation of this
            token, with `text` and `prob` attributes.
        """

        return f'{{text: "{escape_all(self.text)}",prob: {self.prob}}}'


class AlternativeTokens(Visualization):
    """
    `Visualization` of the alternative tokens that the model
    has available at each output token that is picked.
    """

    def __init__(
        self,
        candidate_token_groups: list[list[Token]],
        selected_indices: list[int],
        fallback_tokens: list[Token],
    ):
        """
        Create a new `AlternativeTokens` visualization.

        Args:
            candidate_token_groups (list[list[Token]]): A list where
                each element is a list of `Token`s representing the
                alternative tokens at each output token in the
                response.
            selected_indices (list[int]): A list containing the index
                (starting at 1) for the token that was selected at
                each stage.
            fallback_tokens (list[Token]): A list containing the
                tokens that were selected but were not one of the
                tokens in the `candidate_token_groups`. Should be
                sorted so that the fallback token for the earliest
                group of alternative tokens comes first, the one for
                the second earliest comes second and so on.
        """

        super().__init__()
        self.__candidate_token_groups__ = candidate_token_groups
        self.__selected_indices__ = selected_indices
        self.__fallback_tokens__ = fallback_tokens
        self.__name__ = "Alternative Tokens"
        self.__legend_id__ = "legend_" + str(self.get_uuid())

    def get_html(self) -> str:
        html = '<div style="overflow:auto;">'
        html += f'<canvas id="{self.get_uuid()}" width="1280" height="500">'
        html += "</canvas>"
        html += "</div>"
        html += f'<canvas id="{self.__legend_id__}" width="500", height="110">'
        html += "</canvas>"
        return html

    def get_js(self) -> str:
        return self.call_function(
            "drawAlternativeTokens",
            f'"{self.get_uuid()}"',
            f'"{self.__legend_id__}"',
            js_tools.list_as_js(
                [
                    js_tools.list_as_js(group, do_conversion=True)
                    for group in self.__candidate_token_groups__
                ]
            ),
            self.__selected_indices__,
            js_tools.list_as_js(self.__fallback_tokens__, do_conversion=True),
        )

    def get_dependencies(self):
        return ["js/alternative_tokens.js"]


class TokenSpecificRadarChart(RadarChart):
    """
    Shows a different radar chart depending on the selected token.
    """

    def __init__(self, token_values: list[list[list[any]]]):
        """
        Create a new `TokenSpecificRadarChart`.

        Args:
            token_values (list[list[list[any]]]): 2D list where each
                element is another list where the first element is the
                selected token's text and the second element is a 2D
                list where each element's first element is a candidate
                token's text and the second element is its log
                probability.
        """

        self.__token_values__ = token_values

        super().__init__(token_values[0][1])
        self.__name__ = "Token Specific Radar Chart"
        self.__selector_id__ = "selector_" + str(self.get_uuid())

    def get_html(self):
        html = "<style>"
        html += f".llmvis-token-button"
        html += "{"
        html += "background-color: transparent;"
        html += "border: none;"
        html += "font-size: medium;"
        html += "}"
        html += f".llmvis-token-button.selected"
        html += "{"
        html += "color: rgb(111, 113, 140);"
        html += "}"
        html += "</style>"
        html += f'<div id="{self.__selector_id__}" style="display:table-cell;padding:10px;">'
        html += "</div>"
        html += "<br>"

        return html + super().get_html()

    def get_js(self) -> str:
        return (
            self.call_function(
                "connectButtonsToRadarChart",
                f'"{self.get_uuid()}"',
                f'"{self.__selector_id__}"',
                self.__token_values__,
            )
            + super().get_js()
        )

    def get_dependencies(self):
        return ["js/token_specific_radar_chart.js"] + super().get_dependencies()


class TemperatureSpecificVisualization(Visualization):
    """
    A `Visualization` that changes as an interactive temperature
    control is changed.
    """

    def __init__(
        self,
        slider_values: dict[float, any],
        min_value: float,
        max_value: float,
        step: float,
        js_func: str,
    ):
        """
        Create a new `TemperatureSpecificVisualization`.

        Args:
            slider_values (dict[float, any]): A dictionary mapping
                each temperature value to the data that should
                be visualized at that temperature value.
            min_value (float): The minimum temperature value.
            max_value (float): The maximum temperature value.
            step (float): The step between each temperature value.
            js_func (str): A function that should be called when
                the temperature is changed that takes a single input
                containing the data for the selected temperature
                value according to `slider_values`.
        """

        super().__init__()

        self.__slider_values__ = slider_values
        self.__value__ = min_value
        self.__min_value__ = min_value
        self.__max_value__ = max_value
        self.__step__ = step
        self.__slider_id__ = "slider_" + str(self.get_uuid())
        self.__text_input_id__ = "text_input_" + str(self.get_uuid())
        self.__js_func__ = js_func

    def get_html(self):
        css = f"#{self.__slider_id__}"
        css += "{"
        css += "-webkit-appearance:none;"
        css += "appearance:none;"
        css += "border-radius:10px;"
        css += "outline:none;"
        css += "background:rgb(111, 113, 140);"
        css += "vertical-align: middle;"
        css += "height:50%;"
        css += "}"
        css += f"#{self.__slider_id__}::-webkit-slider-thumb, #{self.__slider_id__}::-moz-range-thumb"
        css += "{"
        css += "-webkit-appearance:none;"
        css += "appearance:none;"
        css += "background:rgb(191, 191, 191);"
        css += "border-color:rgb(191, 191, 191);"
        css += "}"
        css += f"#{self.__text_input_id__}"
        css += "{"
        css += "background-color:transparent;"
        css += "border:none;"
        css += "border-bottom: 1px solid rgb(191,191,191);"
        css += "text-align:center;"
        css += "width:8%;"
        css += "}"

        html = "<style>"
        html += css
        html += "</style>"
        html += '<div style="margin:10px;">'
        html += "<form>"
        html += f'<label for="{self.__slider_id__}" class="llmvis-text" '
        html += 'style="padding-right:20px;">'
        html += "Temperature Value: "
        html += "</label>"
        html += f'<input type="range" id="{self.__slider_id__}" '
        html += f'min="{self.__min_value__}" max="{self.__max_value__}" '
        html += f'step="{self.__step__}" value="{self.__value__}">'
        html += f'<input type="text" id="{self.__text_input_id__}" class="llmvis-text">'
        html += "</form>"
        html += "</div>"

        return html

    def get_js(self):
        return self.call_function(
            "connectTemperatureControlsToVisualization",
            f"{self.__js_func__}",
            f'"{self.__slider_id__}"',
            f'"{self.__text_input_id__}"',
            self.__slider_values__,
        )

    def get_dependencies(self):
        return ["js/temperature_specific.js"]


class TemperatureSpecificRadarChart(TemperatureSpecificVisualization):
    """
    `TokenSpecificRadarChart` that changes as a temperature slider
    is changed.
    """

    def __init__(
        self,
        slider_values: dict[float, list[list[list]]],
        min_value: float,
        max_value: float,
        step: float,
    ):
        """
        Create a new `TemperatureSpecificRadarChart`.

        Args:
            slider_values (dict[float, any]): A dictionary mapping
                each temperature value to the data that should
                be visualized by the `TokenSpecificRadarChart`.
            min_value (float): The minimum temperature value.
            max_value (float): The maximum temperature value.
            step (float): The step between each temperature value.
            js_func (str): A function that should be called when
                the temperature is changed that takes a single input
                containing the data for the selected temperature
                value according to `slider_values`.
        """

        self.__uuid__ = None

        self.__vis__ = TokenSpecificRadarChart(slider_values[min_value])
        canvas_id = self.__vis__.get_uuid()
        selector_id = f"selector_{self.__vis__.get_uuid()}"

        super().__init__(
            slider_values,
            min_value,
            max_value,
            step,
            js_func=f'(val) => connectButtonsToRadarChart("{canvas_id}","{selector_id}",val)',
        )

        self.__name__ = "Temperature Specific Radar Chart"

    def get_html(self):
        return super().get_html() + self.__vis__.get_html()

    def get_js(self):
        return super().get_js() + self.__vis__.get_js()

    def get_dependencies(self):
        return super().get_dependencies() + self.__vis__.get_dependencies()


class TemperatureSpecificAlternativeTokens(TemperatureSpecificVisualization):
    """
    `AlternativeTokens` `Visualization` that changes as a temperature
    slider is changed.
    """

    def __init__(
        self,
        slider_values: dict[float, tuple[list, list, list]],
        min_value: float,
        max_value: float,
        step: float,
    ):
        """
        Create a new `TemperatureSpecificAlternativeTokens` `Visualization`.

        Args:
            slider_values (dict[float, tuple[list, list, list]]): A
                dictionary mapping each temperature value to a tuple
                containing the three lists required by an
                `AlternativeTokens` visualization.
            min_value (float): The minimum temperature value.
            max_value (float): The maximum temperature value.
            step (float): The step between each temperature value.
            js_func (str): A function that should be called when
                the temperature is changed that takes a single input
                containing the data for the selected temperature
                value according to `slider_values`.
        """

        self.__uuid__ = None
        (
            initial_candidate_token_groups,
            initial_selected_indices,
            initial_fallback_tokens,
        ) = slider_values[min_value]
        self.__vis__ = AlternativeTokens(
            initial_candidate_token_groups,
            initial_selected_indices,
            initial_fallback_tokens,
        )
        canvas_id = self.__vis__.get_uuid()
        converted_slider_values = "{"
        for i, key in enumerate(slider_values):
            converted_slider_values += f"{key}:[{js_tools.list_as_js([js_tools.list_as_js(group, do_conversion=True) for group in slider_values[key][0]])},{slider_values[key][1]},{js_tools.list_as_js(slider_values[key][2], do_conversion=True)},]"

            if i < len(slider_values) - 1:
                converted_slider_values += ","

        converted_slider_values += "}"

        super().__init__(
            converted_slider_values,
            min_value,
            max_value,
            step,
            js_func=f'(val) => drawAlternativeTokens("{canvas_id}", "{self.__vis__.__legend_id__}", val[0], val[1], val[2])',
        )

        self.__name__ = "Temperature Specific Alternative Tokens"

    def get_html(self):
        return super().get_html() + self.__vis__.get_html()

    def get_js(self):
        return super().get_js() + self.__vis__.get_js()

    def get_dependencies(self):
        return super().get_dependencies() + self.__vis__.get_dependencies()
