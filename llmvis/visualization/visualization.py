import abc
import re

from llmvis.visualization.linked_files import relative_file_read

def escape_all(string: str) -> str:
    """
    Given a string, return a new string with necessary special characters
    escaped. Used for escaping strings so that they can safely be
    inserted into JavaScript code stored in a string.

    Args:
        string (str): The string that should necessary special characters
            escaped.

    Returns:
        A new string with necessary special characters escaped.
    """
    return re.sub(r'([\'\n"\\])', r'\\\1', string)

class Unit():
    """
    An individual unit that will be shown by the `TextHeatmap`
    visualization. This is commonly a full word, but smaller
    components of words or large clusters of words can be
    units too.
    """

    def __init__(self, text: str, weight: float,
                 details: list[tuple[str, any]]):
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

        js = '{'
        js += f'text:\'{escape_all(self.text)}\','
        js += f'weight:\'{escape_all(str(self.weight))}\','
        js += 'details: ['

        for i, detail in enumerate(self.details):
            js += f'[\'{escape_all(str(detail[0]))}\', \'{escape_all(str(detail[1]))}\']'

            if i < len(self.details) - 1:
                js += ','

        js += ']}'

        return js

class Visualization(abc.ABC):
    WIDTH = 600
    HEIGHT = 300

    """
    Base class for a visualization. Used to define the HTML
    representation of a specific `Visualization` so it can
    be rendered.
    """

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

class TextHeatmap(Visualization):
    """
    A heatmap for a chunk of text where each individual unit
    of the text (e.g. a word in the text) is colored depending
    on a corresponding weight.
    """

    def __init__(self, units: list[Unit]):
        """
        Create a new `TextHeatmap` for a provided list of `Unit`s.

        Args:
            units (list[Unit]): A list of `Unit`s (such as words) that
                make up the chunk of text that should be
                visualized.
        """

        self.__units = units

        max_weight = None
        min_weight = None

        for unit in units:
            if max_weight == None or max_weight < unit.weight:
                max_weight = unit.weight
            
            if min_weight == None or min_weight > unit.weight:
                min_weight = unit.weight

        # We want the scaling to be the same for positive and
        # negative values. This is because if the highest
        # value is 10.0 and the lowest is -0.2 then values
        # of 10.0 and -0.2 would be bright red and bright
        # blue respectively. This gives the indication to the
        # user that the second value is essentially the negative
        # of the first number which is inaccurate. Thus, calculate
        # the absolute min and max values and find which is higher
        # and use this and the negative of this as the max and
        # min values.
        largest_abs = max(abs(max_weight), abs(min_weight))
        self.__max_weight = largest_abs
        self.__min_weight = -largest_abs

    def get_html(self) -> str:
        font_size = 50
        spacing = 10

        html = f'<canvas id="llmvis-heatmap-canvas" width="{self.WIDTH}" height="{self.HEIGHT}">'
        html += '</canvas>'

        return html
    
    def get_js(self):
        js = relative_file_read('js/heatmap.js')

        js += 'units=['

        for i, unit in enumerate(self.__units):
            js += unit.get_js()
            if i < len(self.__units) - 1:
                js += ','
        
        js += '];'

        js += f'minWeight={self.__min_weight};'
        js += f'maxWeight={self.__max_weight};'
        
        js += 'loadFonts().then(function() {'
        js += 'calculateCanvasSize();'
        js += 'drawHeatmap();'
        js += '});'
        return js