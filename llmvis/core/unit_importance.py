class Combinator():
    """
    Manages combinations of 'units' in a prompt. A 'unit' can be
    a word, a token or any smaller division of a prompt and Combinator
    provides tools for iterating through different combinations of
    these units to evaluate the importance of each one.
    """

    # The prompt, but as a list of individual units
    __separated_prompt = []

    # Maps each word to a list of indices in the eventual similarities list, representing
    # which of the similarity values were calculated with each token removed
    __without_map = {}

    def __init__(self, separated_prompt: list[str]):
        """
        Create a new Combinator based on a given list of
        units.

        Args:
            separated_prompt (list[str]): A list of strings
                where each string represents a unit in the
                prompt
        """

        self.__separated_prompt = separated_prompt
    
    def get_combinations(self) -> list[list[str]]:
        """
        Get a list containing each combination of units in the
        separated prompt.

        Args:
            flatten_delimiter (str): The string that should be inserted
                between each word when flattening the separated string
        
        Returns:
            A 2D list containing each combination of unit strings
        """

        combinations = []
        n = len(self.__separated_prompt)

        for i in range(n):
            # (unit could be a word, a token etc.)
            unit = self.__separated_prompt.pop(i)

            if unit not in self.__without_map:
                self.__without_map[unit] = []
            
            self.__without_map[unit].append(i)

            # Combine words list into single string
            combinations.append(self.__separated_prompt.copy())
            self.__separated_prompt.insert(i, unit)

        return combinations
    
    def get_shapley_values(self, similarities: list[float]) -> list[int]:
        """
        Use combination data to calculate shapley values based
        on a list of similarities.

        Args:
            similarities (list[float]): A list containing the similarity to
                the original response for each combination's
                response
        """

        shapley_values = []

        for word in self.__separated_prompt:
            withouts = self.__without_map[word]
            with_word_average = 0
            without_word_average = 0

            for i, similarity in enumerate(similarities):
                if i in withouts:
                    without_word_average += similarity
                else:
                    with_word_average += similarity
            
            with_word_average /= len(similarities) - len(withouts)
            without_word_average /= len(withouts)

            shapley_values.append(with_word_average - without_word_average)
        
        # Normalize values
        m = max(shapley_values)
        
        for i in range(len(shapley_values)):
            shapley_values[i] = shapley_values[i] / m

        return shapley_values