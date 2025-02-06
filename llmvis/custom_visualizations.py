from llmvis.visualization.visualization import LineChart
from llmvis.visualization.linked_files import relative_file_read

class WordSpecificLineChart(LineChart):
    """
    Modified `LineChart` that also includes a text box where the user
    can enter in the specific line chart that they wish to see. For example,
    if we have a different line chart for the frequency of each word as some
    parameter changes, this visualization allows the user to conveniently
    switch between the line charts for all the different words.
    """

    def __init__(self, word_values: dict[str, list[list[any]]], t_values: list[int]):
        """
        Create a new `WordSpecificLineChart` `Visualization`.

        Args:
            word_values (dict[str, list[list[any]]]): A dictionary mapping each word
                to the line chart data. See `LineChart` for information on how this
                data should be formatted.
            t_values (list[int]): A list of integers for the complete temperature
                values that were used for sampling. This is required so that if
                in a given word it is missing a frequency for a given temperature,
                this can be treated as zero in the visualization and thus the line
                chart x axis contains the same values for all words.
        """

        self.__word = list(word_values.keys())[0]
        self.__word_values = word_values
        self.__t_values = t_values
        super().__init__(self.__filled_list(self.__word))

    def get_name(self):
        return 'Word-Specific Line Chart'

    def get_html(self):
        html = '<input type="text" class="llmvis-textbox" placeholder="Enter a word..." '
        html += f'id="llmvis-word-text-field" value="{self.__word}">'
        html += '<br>'
        return html + super().get_html()

    def get_js(self):
        js = relative_file_read('../js/word_specific_line_chart.js')
        js += f'wordValues={self.__values_as_object()};'
        return js + super().get_js()

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

        obj = '{'

        for i, key in enumerate(self.__word_values.keys()):
            obj += key + ':' + str(self.__filled_list(key))

            if i < len(self.__word_values.keys()) - 1:
                obj += ','

        obj += '}'

        return obj

    def __filled_list(self, word: str) -> list[list[any]]:
        """
        Get the list containing raw line chart data for a given
        `word`, with any missing values filled as 0. Useful for
        giving a consistent line chart x-axis between different
        charts.

        Args:
            word (str): The word that should have its raw line
                chart data filled.

        Returns:
            A 2D list containing the original line chart data for
            `word` but with any missing data inserted with a value
            of `0`.
        """

        values = self.__word_values[word]

        # Already contains everything, we can just use it as-is
        if len(values) == len(self.__t_values):
            return values

        i = 0
        filled = []

        for t in self.__t_values:
            if i < len(values) and values[i][0] == t:
                filled.append(values[i])
                i += 1
            else:
                filled.append([t, 0.0])

        return filled