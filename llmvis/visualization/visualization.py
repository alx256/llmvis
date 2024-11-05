import abc

BACKGROUND_RGB_VALUE = 69

class Visualization(abc.ABC):
    """
    Base class for a Visualization. Used to define the HTML
    representation of a specific visualization so it can
    be rendered.
    """

    @abc.abstractmethod
    def get_html(self) -> str:
        """
        Get the HTML representation of this Visualization.

        Returns:
            A string containing the HTML representation of this
                Visualization
        """

        pass

class TextHeatmap(Visualization):
    """
    A heatmap for a chunk of text where each individual unit
    of the text (e.g. a word in the text) is colored depending
    on a corresponding weight.
    """

    def __init__(self, units: list[str], weights: list[float]):
        """
        Create a new TextHeatmap for a provided list of units
        and corresponding list of weights. Note that the units
        and weights lists must be the same length so that there
        is a one-to-one mapping between each unit and its
        weight.

        Args:
            units (list[str]): A list of units (such as words) that
                make up the chunk of text that should be
                visualized
            weights (list[float]): A list of floats corresponding to
                each unit, determining the coloring of it. Each weight
                can be positive or negative with more positive weights
                being hotter and more negative weights being colder.
        """

        self.__units = units
        self.__weights = weights

        max_weight = max(weights)
        min_weight = min(weights)

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
        max_weight = max(weights)
        min_weight = min(weights)
        largest_abs = max(abs(max_weight), abs(min_weight))
        self.__max_weight = largest_abs
        self.__min_weight = -largest_abs

    def get_html(self) -> str:
        html = '<div style = "display: flex; gap: 7px; flex-wrap: wrap; font-size: xx-large;">'

        for i in range(len(self.__units)):
            unit = self.__units[i]
            weight = self.__weights[i]

            rgb = self.__calculate_rgb(weight)

            # Represent each word as <div> so it can be colored independently
            html += f'<div class = "llmvis-text" style = "background-color: {rgb};">' + unit + '</div>'

        rgb_start = self.__calculate_rgb(self.__min_weight)
        rgb_mid = self.__calculate_rgb(0.0)
        rgb_end = self.__calculate_rgb(self.__max_weight)

        html += '</div>'
        # Key
        html += '<div style="margin: 25px">'
        html += f'<div style="background-image: linear-gradient(to right, {rgb_start}, {rgb_mid}, {rgb_end}); padding: 9px;"></div>'
        html += f'<div class="llmvis-text" style="float: left;">{self.__min_weight} (Least Important)</div>'
        html += f'<div class="llmvis-text" style="float: right;">{self.__max_weight} (Most Important)</div>'
        html += '</div>'

        return html
    
    def __calculate_rgb(self, weight: float) -> str:
        """
        Calculate the corresponding RGB values that should be
        used for a given weight.

        Args:
            weight (float): The weight that will be used to
                calculate the RGB values
        
        Returns:
            A CSS string representation of this weight's RGB
                values
        """

        rgb = (0.0, 0.0, 0.0)

        # Values near 0 should be closer to white while values
        # near the max or min weights should be closer to red
        # or to blue respectively. For RGB values this is done
        # by keeping red/blue as the max (1.0) and moving the
        # other values away from 1.0 accordingly.
        if weight < 0.0:
            # Move from white to blue
            other_vals = weight / self.__min_weight
            rgb_value = BACKGROUND_RGB_VALUE + ((255 - BACKGROUND_RGB_VALUE) * other_vals)

            rgb = (rgb_value - (rgb_value * other_vals),
                   rgb_value - (rgb_value * other_vals),
                   rgb_value)
        else:
            # Move from white to red
            other_vals = weight / self.__max_weight
            rgb_value = BACKGROUND_RGB_VALUE + ((255 - BACKGROUND_RGB_VALUE) * other_vals)

            rgb = (rgb_value,
                   rgb_value - (rgb_value * other_vals),
                   rgb_value - (rgb_value * other_vals))

        return f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'