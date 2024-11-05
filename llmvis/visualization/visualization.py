import abc

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
            rgb_str = f'rgb({rgb[0] * 255}, {rgb[1] * 255}, {rgb[2] * 255})'

            # Represent each word as <div> so it can be colored independently
            html += f'<div style = "background-color: {rgb_str};">' + unit + '</div>'

        html += '</div>'

        return html
    
    def __calculate_rgb(self, weight: float) -> tuple[float, float, float]:
        """
        Calculate the corresponding RGB values that should be
        used for a given weight.

        Args:
            weight (float): The weight that will be used to
                calculate the RGB values
        
        Returns:
            A tuple with 3 values for the red, green and blue values in the
                range [0, 1]
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
            rgb = (1.0 - other_vals, 1.0 - other_vals, 1.0)
        else:
            # Move from white to red
            other_vals = weight / self.__max_weight
            rgb = (1.0, 1.0 - other_vals, 1.0 - other_vals)

        return rgb