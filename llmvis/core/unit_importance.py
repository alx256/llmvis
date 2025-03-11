import random

from numpy.typing import ArrayLike


class Combinator:
    """
    Manages combinations of 'units' in a prompt. A 'unit' can be
    a word, a token or any smaller division of a prompt and Combinator
    provides tools for iterating through different combinations of
    these units to evaluate the importance of each one.
    """

    def __init__(self, separated_prompt: list[str]):
        """
        Create a new Combinator based on a given list of
        units.

        Args:
            separated_prompt (list[str]): A list of strings
                where each string represents a unit in the
                prompt
        """

        # The prompt, but as a list of individual units
        self.__separated_prompt = separated_prompt
        # Maps each word to a list of indices in the eventual similarities list, representing
        # which of the similarity values were calculated with each token removed
        self.__without_map = {}
        self.__missing_terms__ = []
        self.__length__ = 0

    def get_combinations(self, r: float) -> list[list[str]]:
        """
        Get a list containing each combination of units in the
        separated prompt.

        Args:
            flatten_delimiter (str): The string that should be inserted
                between each word when flattening the separated string

        Returns:
            A 2D list containing each combination of unit strings
        """

        # Essential set
        E = []
        # Combinations set
        C = []
        n = len(self.__separated_prompt)
        sampled_combinations = int((2**n - 1) * r)
        additional_samples = max(0, sampled_combinations - n)

        for i in range(n):
            # (unit could be a word, a token etc.)
            unit = self.__separated_prompt.pop(i)

            self.__without_map.setdefault(unit, [])
            self.__without_map[unit].append(i)
            self.__missing_terms__.append([i])

            # Combine words list into single string
            E.append(self.__separated_prompt.copy())
            self.__separated_prompt.insert(i, unit)

        # Random samples set
        S = self.__random_samples__(additional_samples, n)
        C = E + S
        self.__length__ = len(C)

        return C

    def get_shapley_values(self, similarities: list[float]) -> list[int]:
        """
        Use combination data to calculate shapley values based
        on a list of similarities.

        Args:
            similarities (list[float]): A list containing the similarity to
                the original response for each combination's
                response
        """

        assert len(similarities) == self.__length__
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

        return self.__normalize_shapley_values__(shapley_values)

    def __random_samples__(self, count: int, n: int) -> list[list[str]]:
        """
        Generate some random combination samples.

        Args:
            count (int): The number of random samples to be generated.
            n (int): The number of units in the separated prompt.

        Returns:
            A list of random combination samples.
        """

        samples = []
        used_samples = []

        for i in range(count):
            indices = self.__get_random_index_order__(n)

            while indices in used_samples:
                indices = self.__get_random_index_order__(n)

            modified_prompt = []

            for j in range(len(self.__separated_prompt)):
                if j not in indices:
                    unit = self.__separated_prompt[j]
                    modified_prompt.append(unit)
                    self.__without_map[unit].append(n + i)

            self.__missing_terms__.append(indices)
            used_samples.append(indices)
            samples.append(modified_prompt)

        return samples

    def get_missing_terms(self, index: int) -> ArrayLike:
        """
        Get an `ArrayLike` containing the indices in the original
        separated prompt that have been removed for a given combination
        index.

        Args:
            index (int): The combination index that is being queried.
                This is the index of the combination in the list
                returned by `get_combinations`.

        Returns:
            An `ArrayLike` containing the indices in the original
            separated prompt that have been removed.
        """

        return self.__missing_terms__[index]

    def __get_random_index_order__(self, n: int) -> set[int]:
        indices = set()
        # If we only remove 1 unit then this sample will be in
        # E and therefore already considered so we must remove
        # at least 2. If we remove all units then the result
        # will be empty and useless so we must at least keep one
        # remaining
        removal_count = random.randint(2, n - 1)

        while len(indices) < removal_count:
            # Find a random index
            indices.add(random.randint(0, n - 1))

        return indices

    def __normalize_shapley_values__(
        self, shapley_values: list[float], power: int = 1
    ) -> list[float]:
        """
        Normalize the shapley values.
        Modified from https://github.com/ronigold/TokenSHAP/blob/main/token_shap/token_shap.py

        Args:
            shapley_values (list[float]): The shapley values that
                should be normalized.
            power (int): The power that should be used for the
                normalization. Default is `1`.

        Returns:
            A list of normalized Shapley values.
        """
        min_value = min(shapley_values)
        shifted_values = [val - min_value for val in shapley_values]
        powered_values = [val**power for val in shifted_values]
        total = sum(powered_values)
        if total == 0:
            return [1 / len(powered_values) for _ in powered_values]
        return [val / total for val in powered_values]
