from __future__ import annotations
import abc
import asyncio
import math
import re
import ollama
import numpy as np
from typing_extensions import Optional
import json
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from llmvis.core.unit_importance import Combinator
from llmvis.visualization import Visualizer
from llmvis.visualization.visualization import (
    HeatmapColorScheme,
    Point,
    Unit,
    TextHeatmap,
    TableHeatmap,
    TagCloud,
    ScatterPlot,
    BarChart,
    LineChart,
)
from llmvis.core.preprocess import should_include
from llmvis.custom_visualizations import (
    AlternativeTokens,
    TemperatureSpecificRadarChart,
    Token,
    TokenSpecificRadarChart,
    WordSpecificLineChart,
    AIClassifier,
    TemperatureSpecificAlternativeTokens,
)

MEDIATION_PROMPT = """You will be given some JSON data, containing a numerical value followed by a string. You must perform two tasks on the data you are given.

First task: You must establish a number of classes for the data based on the text items. Try to find similar concepts and ideas within the text items to establish as few classes as possible.  Then, using these classes you have created, classify each numerical item into one of the classes.

Second task: You must determine the number of hallucinations associated with each numerical value based on its associated text item. A hallucination is defined as an incorrect statement.

IMPORTANT DETAILS:
You must only return a JSON string. Nothing more.
Only the numerical values in the input must be classified.
All of the inputted numerical values must be classified. Make sure that each one is assigned to a class.

Examples are shown below:
Input: {
\"input\": [
[0.1, \"It is sunny today\"],
[0.2, \"The time is currently 12:00\"],
[0.3, \"It rained last week\"],
[0.4, \"Tomorrow there is meant to be snow\"]
]
}
Output: {
\"classes\":[
[\"Statements about the weather\", [0.1, 0.3, 0.4]],
[\"Statements about the time\", [0.2]]
],
\"hallucinations\":[
{\"t\": 0.1, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.2, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.3, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.4, \"count\": 0, \"explanation\": \"No hallucinations detected.\"}
]
}
Input: {
\"input\": [
[0.843924, \"Hamburgers are made with pork\"],
[0.3192381293, \"Football has different meanings in American and British English\"]
[0.289178237, \"Vegetables are good for you but candy is better!\"],
[0.9393939, \"Coca-Cola is an Spanish brand. It was created in 1989 by Jorge Cola.\"],
[0.42938293, \"Water is a hydrating beverage\"],
[0.9923231, \"Swimming is great for fitness\"],
[0.232819912, \"Chicken is rich in protein\"]
]
}
Output: {
\"classes\":[
[\"Food\", [0.843924, 0.289178237, 0.232819912]],
[\"Drink\", [0.9393939, 0.42938293]],
[\"Sports\", [0.3192381293, 0.9923231]]
],
\"hallucinations\":[
{\"t\": 0.843924, \"count\": 1, \"explanation\": \"Hamburgers are made with beef.\"},
{\"t\": 0.3192381293, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.289178237, \"count\": 1, \"explanation\": \"Candy is not healthier than vegetables.\"},
{\"t\": 0.9393939, \"count\": 2, \"explanation\": \"Coca-Cola is an American brand. It was created in 1886 by John Pemberton.\"},
{\"t\": 0.42938293, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.9923231, \"count\": 0, \"explanation\": \"No hallucinations detected.\"},
{\"t\": 0.232819912, \"count\": 0, \"explanation\": \"No hallucinations detected.\"}
]
}"""

MEDIATION_ATTEMPTS = 5

# IDs for metrics
# Used for tracking what the last executed metric
# was.
WORD_IMPORTANCE_GEN_SHAPLEY = 0
WORD_IMPORTANCE_EMBED_SHAPLEY = 1
K_TEMPERATURE_SAMPLING = 2


class ImportanceMetric:
    """
    The approach that should be used for calculating the
    importance of a `Unit`. Available options:

    - **INVERSE_COSINE** - `1.0 - {calculated cosine similarity}`
    - **SHAPLEY** - Game theory Shapley value
    """

    INVERSE_COSINE = "Inverse Cosine Similarity"
    SHAPLEY = "Shapley Value"


class UnitType:
    """
    A type of unit. A "unit" is defined as a component of a prompt.
    Available options:

    - **TOKEN** - A token, representing a word or a smaller part of
    a word determined by a tokenization algorithm.
    - **WORD** - A word, defined as a sequence of letters surrounded
    by whitespace.
    - **SENTENCE** - A sentence, defined as a sequence of words
    surrounded by punctuation that can terminate a sentence
    (e.g. `.`, `!` or `?`).
    """

    TOKEN = 0
    WORD = 1
    SENTENCE = 2


class ImportanceCalculation:
    """
    An approach for calculating importance. Available options:

    - **GENERATION** - Importance should be calculated by
        generating responses and calculating the TF-IDF vector.
    - **EMBEDDING** - Importance should be calculated by
        generating the embedding vector.
    """

    GENERATION = 0
    EMBEDDING = 1


class ModelResponse:
    """
    Object to contain a response that a model can give,
    standardised between all `Connection`s.
    """

    def __init__(self, message: str):
        """
        Create a new `ModelResponse`.

        Args:
            message (str): The message that was given
                by the model.
        """

        self.message = message
        self.candidate_token_groups = []
        self.selected_indices = []
        self.fallback_tokens = []
        self.alternative_tokens = []
        self.prob_sum = 0
        self.generated_tokens_count = 0


class Connection(abc.ABC):
    """
    Base class for connections. A 'connection' is a link between the program
    written by the user and the service that is used for hosting the LLMs.
    Override this class to add support for LLMVis integrating with another
    service.
    """

    __last_metric_id__ = -1
    __last_metric_data__ = {}

    def token_importance_gen(self, prompt: str):
        """
        Calculate the token importance of a given prompt by
        by how each token impacts the generated text and
        display a visualization of this. While this is a
        very accurate measure of token importance, this
        method can use a lot of input tokens so if
        you are using a cloud platform that charges based on
        input tokens used, then consider alternative token
        importance approaches. It can also be slow if you are
        running a model locally.

        Args:
            prompt (str): The prompt that should have its token
                importance calculated.
        """

        # TODO: Implement when this pull request is approved: https://github.com/ollama/ollama/pull/6586

    def unit_importance(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        importance_metric: ImportanceMetric = ImportanceMetric.INVERSE_COSINE,
        calculation: ImportanceCalculation = ImportanceCalculation.GENERATION,
        unit_type: UnitType = UnitType.WORD,
        sampling_ratio: float = 0.0,
        use_perplexity_difference: bool = False,
        test_system_prompt: bool = False,
    ) -> Visualizer:
        """
        Calculate the importance of each word in a given prompt
        using the TokenSHAP algorithm (see https://arxiv.org/html/2407.10114v2)
        and get a `Visualizer` visualizing this of this.

        Args:
            prompt (str): The prompt that should have its
                word importance calculated
            system_prompt (Optional[str]): The system prompt that
                should be used in generating responses. To not use
                a system prompt use `None`. Default is `None`.
            importance_metric (ImportanceMetric): The `ImportanceMetric`
                that should be used to determine importance. Default is
                `ImportanceMetric.INVERSE_COSINE`.
            calculation (ImportanceCalculation): The `ImportanceCalculation`
                that should be used for calculating vectors for similarities.
                Default is `ImportanceCalculation.GENERATION`.
            unit_type (UnitType): The `UnitType` that should have the
                importance calculated for. Default is `UnitType.WORD`.
            sampling_ratio (float): How many random samples should
                be carried out (0.0 for none)
            use_perplexity_difference (bool): Set this to `True` to
                enable perplexity difference calculations for
                hallucination detection.
            test_system_prompt (bool): Set this to `True` to calculate
                the word importance for the system prompt instead of
                the main prompt.

        Returns:
            A `Visualizer` showing a table heatmap, a text heatmap and a tag cloud
            for the importance of each word.
        """

        if importance_metric != ImportanceMetric.SHAPLEY and sampling_ratio != 0.0:
            raise RuntimeError(
                "Sampling ratio is only supported with the Shapley similary index"
            )

        if test_system_prompt and system_prompt is None:
            raise RuntimeError(
                "Cannot test the system prompt if no system prompt is provided!"
            )

        if (
            use_perplexity_difference
            and calculation != ImportanceCalculation.GENERATION
        ):
            raise RuntimeError(
                "Perplexity difference is only support with the generation calculation"
            )

        test_prompt = prompt if not test_system_prompt else system_prompt
        separated_prompt = []

        if unit_type == UnitType.TOKEN:
            # TODO: Implement when this pull request is approved: https://github.com/ollama/ollama/pull/6586
            raise RuntimeError("Token importance is currently not supported")
        elif unit_type == UnitType.WORD:
            # Nothing fancy needed for 'tokenizing' in terms of words, only splitting by spaces
            separated_prompt = test_prompt.split(" ")
        elif unit_type == UnitType.SENTENCE:
            punctuation = "?.!"
            # This regex expression adds empty strings by design,
            # so filter them out.
            separated_prompt = list(
                filter(
                    None, re.split(f"([^{punctuation}]*[{punctuation}]+)", test_prompt)
                )
            )

        combinator = Combinator(separated_prompt)
        combinations = combinator.get_combinations(r=sampling_ratio)
        requests = [test_prompt] + [
            self.__flatten_words(combination, delimiter=" ")
            for combination in combinations
        ]
        missing_terms_strs = []
        vectors = []

        if calculation == ImportanceCalculation.GENERATION:
            vectors, responses, outputs = self.__batch_generate__(
                system_prompt, use_perplexity_difference, requests
            )
        elif calculation == ImportanceCalculation.EMBEDDING:
            vectors = np.array(self.__calculate_embeddings__(requests))
            responses = ["N/A"] * len(requests)
            outputs = vectors

        for i in range(len(requests[1:])):
            missing_terms_strs.append("")
            missing_term_indices = combinator.get_missing_terms(i)

            for index in missing_term_indices:
                missing_term = separated_prompt[index]
                combinations[i].insert(index, "_" * len(missing_term)), " "
                requests[i + 1] = self.__flatten_words(combinations[i], delimiter=" ")

                missing_terms_strs[-1] += missing_term

                if i < len(missing_term_indices) - 1:
                    missing_terms_strs[-1] += ", "

        # Use TF-IDF representation to calculate similarity between each
        # response and the full response
        similarities = cosine_similarity(
            vectors[0].reshape(1, -1), vectors[1:]
        ).flatten()

        # Start the visualization
        vals = []

        if importance_metric == ImportanceMetric.INVERSE_COSINE:
            vals = [1.0 - similarity for similarity in similarities]
        elif importance_metric == ImportanceMetric.SHAPLEY:
            vals = combinator.get_shapley_values(similarities)
        else:
            raise RuntimeError("Invalid similarity index used!")

        importance_units = []
        full_prompt_perplexity = (
            self.__perplexity__(
                responses[0].generated_tokens_count, responses[0].prob_sum
            )
            if use_perplexity_difference
            else 0.0
        )
        perplexity_difference_units = []
        table_contents = []

        for i in range(len(separated_prompt)):
            text = separated_prompt[i]
            val = vals[i]
            response = responses[i + 1]

            # Each unit's weight represents the shapley
            # (importance) value of that unit.
            importance_units.append(
                Unit(
                    text,
                    val,
                    (
                        [
                            (importance_metric, val),
                        ]
                        + (
                            [("Generated Prompt", response.message)]
                            if calculation == ImportanceCalculation.GENERATION
                            else []
                        )
                    ),
                )
            )

            if use_perplexity_difference:
                perplexity = self.__perplexity__(
                    response.generated_tokens_count, response.prob_sum
                )
                # Positive difference indicates rise in perplexity,
                # negative indicates fall.
                perplexity_difference = perplexity - full_prompt_perplexity

                # Each unit's weight represents the perplexity
                # (hallucination likelihood) difference of that unit.
                perplexity_difference_units.append(
                    Unit(
                        text,
                        perplexity_difference,
                        [("Perplexity Difference", perplexity_difference)],
                    )
                )

        table_contents = [
            [
                requests[i],
                outputs[i],
                missing_terms_strs[i - 1] if i > 0 else "N/A",
                # str(shapley_vals[i - 1]) if i > 0 else "N/A",
                str(similarities[i - 1]) if i > 0 else "N/A",
            ]
            for i in range(len(outputs))
        ]

        table_heatmap = TableHeatmap(
            contents=table_contents,
            headers=[
                "Prompt",
                "Model Response",
                "Missing Term",
                # "Shapley Value",
                "Cosine Difference",
            ],
            weights=[0.0] + vals,
        )
        table_heatmap.set_comments(self.__get_info__())

        additional_args = {}

        if importance_metric == ImportanceMetric.INVERSE_COSINE:
            additional_args["min_value"] = 0.0
            additional_args["max_value"] = 1.0

        importance_metric_comment = "Importance Metric: " + importance_metric

        text_heatmap = TextHeatmap(importance_units, **additional_args)
        text_heatmap.set_comments(self.__get_info__(), importance_metric_comment)

        tag_cloud = TagCloud(importance_units)
        tag_cloud.set_comments(self.__get_info__(), importance_metric_comment)

        visualizations = [table_heatmap, text_heatmap, tag_cloud]

        if use_perplexity_difference:
            perplexity_difference_heatmap = TextHeatmap(
                perplexity_difference_units, color_scheme=HeatmapColorScheme.GREEN_RED
            )
            perplexity_difference_heatmap.set_name("Perplexity Difference Text Heatmap")
            perplexity_difference_heatmap.set_comments(self.__get_info__())
            visualizations.append(perplexity_difference_heatmap)

        if calculation == ImportanceCalculation.EMBEDDING:
            # Use PCA to reduce dimensionality to suppress noise and speed up
            # computations (per scikit-learn recommendations
            # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(vectors)

            scatter_plot = ScatterPlot(
                [
                    Point(
                        reduced[i][0],
                        reduced[i][1],
                        "Missing terms: "
                        + (
                            "No missing terms"
                            if i == 0
                            else " ".join(
                                [
                                    separated_prompt[j]
                                    for j in combinator.get_missing_terms(i - 1)
                                ]
                            )
                        ),
                    )
                    for i in range(len(reduced))
                ]
            )
            scatter_plot.set_comments(self.__get_info__())
            visualizations.append(scatter_plot)

        return Visualizer(visualizations)

    def k_temperature_sampling(
        self,
        prompt: str,
        k: int,
        system_prompt: Optional[str] = None,
        start: float = 0.0,
        end: float = 1.0,
        alternative_tokens: bool = False,
    ) -> Visualizer:
        """
        Sample `k` temperature values starting with `start` and ending
        with `end` to examine the differences between temperature
        values and get a `Visualizer` visualizing this.

        Args:
            prompt (str): The prompt that should be used with the
                different temperature values.
            k (int): The number of samples that should be used. Must
                be greater than 2.
            system_prompt (Optional[str]): The system prompt that
                should be used for generations. Can be `None` to
                not use any system prompt. Default is `None`.
            start (float): The starting temperature value. Default
                is 0.0. Must be less than `end`.
            end (float): The ending temperature value. Default is
                1.0. Must be greater than `start`.
            alternative_tokens (bool): Set this to `True` to enable
                visualizations that use alternative tokens and their
                log probabilities. Only supported on some connections.
                Default is `False`.

        Returns:
            A `Visualizer` showing a table containing the results of the
            different samples.
        """

        if k < 2:
            raise RuntimeError("k must be at least 2.")

        if end < start:
            raise RuntimeError("start must come before the end.")

        if end == start:
            raise RuntimeError("start and end cannot be the same.")

        step = (start + end) / (k - 1)
        t = start
        # Temperature values
        temperatures = []
        # Store the frequency of each word to visualize as a bar chart
        frequencies = {}
        # Store the frequencies as temperatures change to visualize as a line chart
        temperature_change_frequencies = {}

        samples = []
        alternative_tokens_data = {}
        radar_chart_data = {}
        t_values = []

        for _ in range(k):
            t_values.append(t)
            sample = self.__make_request__(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=t,
                alternative_tokens=alternative_tokens,
            )

            samples.append([t, sample.message])
            temperatures.append(t)

            if alternative_tokens:
                alternative_tokens_data[t] = [
                    sample.candidate_token_groups,
                    sample.selected_indices,
                    sample.fallback_tokens,
                ]
                radar_chart_data[t] = sample.alternative_tokens

            for word in sample.message.split(" "):
                # Only keep alpha-numeric (i.e. ignore punctuation) chars
                # of the string
                chars = "".join(e.lower() for e in word if e.isalnum())

                # Edge case where a "word" might contain no alphanumeric characters
                # for example just a dash '-' surrounded by spaces
                if len(chars) == 0:
                    continue

                frequencies.setdefault(chars, 0)
                frequencies[chars] += 1
                temperature_change_frequencies.setdefault(chars, [])

                # Calculate the frequencies for each temperature value for this word
                if (
                    len(temperature_change_frequencies[chars]) > 0
                    and temperature_change_frequencies[chars][-1].x == t
                ):
                    temperature_change_frequencies[chars][-1].y += 1
                else:
                    temperature_change_frequencies[chars].append(Point(t, 1))

            t += step

        table_heatmap = TableHeatmap(
            # TODO: Add original rounding approach back
            contents=[[("{0:.2f}".format(t)), sample] for t, sample in samples],
            headers=["Sampled Temperature", "Sample Result"],
        )
        table_heatmap.set_comments(self.__get_info__())

        # Final frequencies list in expected format for bar chart
        frequencies_list = []

        for name in frequencies.keys():
            # Only keep interesting words
            if should_include(name):
                frequencies_list.append([name, frequencies[name]])

        # Sort in ascending order
        frequencies_list = sorted(
            frequencies_list, key=lambda entry: entry[1], reverse=True
        )
        # Don't want too many bars on the bar chart
        frequencies_list = frequencies_list[:7]
        bar_chart = BarChart(
            frequencies_list, x_axis_label="Word", y_axis_label="Frequency"
        )
        bar_chart.set_comments(self.__get_info__())

        line_chart = WordSpecificLineChart(temperature_change_frequencies, t_values)
        line_chart.set_comments(self.__get_info__())

        Connection.__last_metric_id__ = K_TEMPERATURE_SAMPLING
        Connection.__last_metric_data__ = {
            "samples": samples,
            "temperatures": temperatures,
        }

        visualizations = [table_heatmap, bar_chart, line_chart]

        if alternative_tokens:
            visualizations.append(
                TemperatureSpecificAlternativeTokens(
                    alternative_tokens_data, start, end, step
                )
            )
            visualizations.append(
                TemperatureSpecificRadarChart(radar_chart_data, start, end, step)
            )

        return Visualizer(visualizations)

    def sandbox(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: int = 0.7
    ) -> Visualizer:
        """
        "Sandbox" metric. This can be used to experiment how different prompts and
        parameters affect the output.

        Args:
            prompt (str): The prompt that this sandbox metric should explore.
            system_prompt (Optional[str]): The system prompt that should be
                used for generations in this sandbox metric.
            temperature (int): The temperature that responses should use.
                Default is 0.7.
        """
        model_response = self.__make_request__(
            prompt, system_prompt, temperature, alternative_tokens=True
        )
        temperature_comment = f"Temperature: {temperature}"

        alternative_tokens = AlternativeTokens(
            model_response.candidate_token_groups,
            model_response.selected_indices,
            model_response.fallback_tokens,
        )
        alternative_tokens.set_comments(
            self.__get_info__(),
            temperature_comment,
        )

        radar_chart = TokenSpecificRadarChart(model_response.alternative_tokens)
        radar_chart.set_comments(self.__get_info__(), temperature_comment)

        return Visualizer([alternative_tokens, radar_chart])

    def ai_analytics(self) -> Visualizer:
        """
        Get a `Visualizer` containing relevant additional AI-generated
        analytics for the metric that was run most recently, provided that
        it is supported.

        #### Supported Metrics
        - **K Temperature Sampling**: Can be used to show an AI-generated
            graph classifying each temperature value into a number of
            classes and AI-generated hallucination detection.

        Returns:
            A `Visualizer` that can be used to visualize relevant additional
            AI-generated analytics regarding the most recently run metric.

        Raises:
            RuntimeError: Raised if this is being executed when the most recent
                metric does not support AI analytics or no metric was run just
                before calling this.
        """

        if Connection.__last_metric_id__ == -1:
            raise RuntimeError("Cannnot show AI analytics if no metrics have been run!")

        if Connection.__last_metric_id__ == K_TEMPERATURE_SAMPLING:
            return self.__k_temperature_sampling_ai_analytics__()

        raise RuntimeError(
            "Tried to show AI analytics for a metric that does not support this!"
        )

    def __k_temperature_sampling_ai_analytics__(self) -> Visualizer:
        """
        AI Analytics for K Temperature Sampling. Generates visualizations for
        classifying temperature values into common classes as well as LLM-powered
        hallucination detection.

        Returns:
            A `Visualizer` that can be used to visualize the additional
            visualizations.
        """

        samples = Connection.__last_metric_data__["samples"]
        temperatures = Connection.__last_metric_data__["temperatures"]
        response_data = None
        attempts = 0
        t = 0.0

        while response_data is None:
            if attempts >= MEDIATION_ATTEMPTS:
                break

            response_data = self.__mediate__(
                prompt=MEDIATION_PROMPT,
                data=samples,
                format={
                    "type": "object",
                    "properties": {
                        "classes": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "number"}},
                                ],
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "hallucinations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "t": {"type": "number"},
                                    "count": {"type": "number"},
                                    "explanation": {"type": "string"},
                                },
                            },
                            "minItems": len(samples),
                            "maxItems": len(samples),
                        },
                    },
                },
                temperature=t,
            )
            attempts += 1
            t += 1.0 / MEDIATION_ATTEMPTS

        visualizations = []

        try:
            ai_classifier = AIClassifier(
                (response_data["classes"] if "classes" in response_data else []),
                temperatures,
            )
            ai_classifier.set_comments(self.__get_info__())
            visualizations.append(ai_classifier)
        except KeyError:
            # TODO: Error
            pass

        try:
            if len(response_data["hallucinations"]) >= 2:
                hallucinations_line_chart = LineChart(
                    [
                        Point(r["t"], r["count"], r["explanation"])
                        for r in response_data["hallucinations"]
                    ]
                    if "hallucinations" in response_data
                    else []
                )
                hallucinations_line_chart.set_comments(self.__get_info__())
                hallucinations_line_chart.set_name("Hallucinations Line Chart")
                visualizations.append(hallucinations_line_chart)
        except KeyError:
            # TODO: Error
            pass

        return Visualizer(visualizations)

    def __flatten_words(self, words: list[str], delimiter: str) -> str:
        """
        Flatten a list of unit strings into a singular string
        separated by a delimiter.

        Args:
            words (list[str]): The list of unit strings
            delimiter (str): The delimiter that should be
                used for separating units (can just be empty)
        """

        words_str = ""
        for i, word in enumerate(words):
            words_str += word

            # Add a separator between each word and the next
            if i < len(words) - 1:
                words_str += delimiter

        return words_str

    def __perplexity__(self, N: int, prob_sum: float) -> float:
        """
        Calculate the *perplexity* of a model output, to determine
        the likelihood of hallucinations in the response.

        Args:
            N (int): The number of output tokens in the output's response.
            prob_sum (float): The sum of all log probabilities in the output.

        Returns:
            A real-valued number representing the perplexity of the response.
        """
        return math.exp(-(1 / N) * prob_sum)

    def __batch_generate__(
        self,
        system_prompt: str,
        alternative_tokens: bool,
        requests: list[str],
    ) -> list[list[float]]:
        """
        Calculate the TF-IDF vectors for a batch of requests.

        Args:
            system_prompt (str): The system prompt that should be
                used for making requests.
            alternative_tokens (bool): Set this to `True` to get
                alternative tokens data (if supported).
            requests (list[str]): A list of strings containing the
                requests that should be made.

        Returns:
            A 2D list where each element is a list of floats
            representing the TF-IDF vector for each request.
        """

        responses = []
        outputs = []

        for i, combination in enumerate(requests):
            request = self.__flatten_words(combination, delimiter=" ")
            response = self.__make_request__(
                request,
                system_prompt,
                temperature=0.0,
                alternative_tokens=alternative_tokens,
            )

            responses.append(response)
            outputs.append(response.message)

        # Calculate TF-IDF representation of each response
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(outputs).toarray()

        return vectors, responses, outputs

    @abc.abstractmethod
    def __get_info__(self):
        """
        Get information about this `Connection`. This should ideally
        contain the name of the model that this `Connection` is using,
        alongside the medium for doing this and any additional information.

        Returns:
        A string containing information about this `Connection`.
        """

        pass

    @abc.abstractmethod
    def __make_request__(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: int,
        alternative_tokens: bool = False,
    ) -> ModelResponse:
        """
        Make a request to this connection using a prompt.

        Args:
            prompt (str): The prompt to be given to the connected model
            system_prompt (Optional[str]): The system prompt that should
                be used for this request. Can be `None` to not use any
                system prompt. Default is `None`.
            temperature (int): The temperature that should be used for
                generation. 0.0 means deterministic behaviour while higher
                temperatures introduce more nondeterminism.
            alternative_tokens (bool): Whether or not this request should
                return alternative tokens for each generated token and each
                token's associated probability. Note that this only works on
                supported `Connection`s, trying this with unsupported
                `Connection`s will raise a `RuntimeError`.

        Returns:
            The model's generated response to the prompt
        """
        pass

    @abc.abstractmethod
    def __calculate_embeddings__(self, prompts: list[str]) -> list[list[float]]:
        """
        Calculate the embeddings for a given list of prompts.

        Args:
            prompts (list[str]): The prompts that should be given to
                the connected model to calculate the embeddings for

        Returns:
            A 2D list of floats containing the embeddings for each prompt.
        """

        pass

    @abc.abstractmethod
    def __mediate__(
        self, prompt: str, data: list[any], format: dict[any, any], temperature: int
    ) -> Optional[dict[any, any]]:
        """
        Use this connection to "mediate" some data. This executes a more complex
        NLP-related task using this model and is used for tasks such as
        classification and hallucination detection.

        Args:
            prompt (str): The prompt detailing what task the model should complete
                on the provided data.
            data (list[list[any]]): The data that should be used for mediation, following
                the format detailed in the prompt.
            format (dict[any, any]): The JSON Schema format that the output should provide.
                May be ignored if underlying connection does not have support for structured
                outputs.
            temperature (int): The temperature that should be used for calculating the
                response.

        Returns:
            A dictionary containing the output data from this or `None` if an issue occured.
        """

        pass


class OllamaConnection(Connection):
    def __init__(self, model_name: str):
        """
        Create a new OllamaConnection using the provided model_name.

        Args:
            model_name (str): The name of the model that is being used
                by the connection. Note that this must be a valid Ollama
                model, otherwise a RuntimeError will be thrown.
        """

        self._model_name = model_name

        # Try to pull the model
        if ollama.pull(model_name)["status"] != "success":
            raise RuntimeError(
                model_name
                + " was not able to be pulled. Check that it is a supported model."
            )

        super().__init__()

    def __get_info__(self):
        return "Model: " + self._model_name + " (through Ollama)"

    def __make_request__(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: int,
        alternative_tokens: bool = False,
    ) -> ModelResponse:
        if alternative_tokens:
            raise RuntimeError(
                "OllamaConnection does not support returning alternative tokens. Try with another connection instead."
            )

        # Additional parameters that may or may not be included depending on
        # what the user has provided.
        additional_params = {}

        if system_prompt is not None:
            additional_params["system"] = system_prompt

        response = ollama.generate(
            model=self._model_name,
            prompt=prompt,
            options={"temperature": temperature},
            **additional_params,
        )

        return ModelResponse(message=response.response)

    def __calculate_embeddings__(self, prompts: list[str]) -> list[list[float]]:
        return ollama.embed(model=self._model_name, input=prompts).embeddings

    def __mediate__(
        self, prompt: str, data: list[any], format: dict[any, any], temperature: int
    ) -> Optional[dict[any, any]]:
        try:
            response = ollama.generate(
                model=self._model_name,
                prompt=json.dumps({"input", data}),
                system=prompt,
                options={"temperature": temperature},
                format=format,
            ).response
            return json.loads(response)
        except json.decoder.JSONDecodeError:
            # LLM gave a response that was unexpected as it did not only contain
            # JSON data.

            return None


class WatsonXConnection(Connection):
    """
    Connection to IBM's watsonx.ai service. **Requires both an IBM API Key and a project
    ID for a watsonx.ai project.** Note that using this connection will consume tokens
    on the API's end.
    """

    def __init__(self, api_key: str, project_id: str, model_name: str, location: str):
        """
        Create a new `WatsonXConnection`. Requires an IBM key and a project ID for a
        watsonx.ai project.

        Args:
            api_key (str): The IBM API key that will be used for connecting to the
                watsonx.ai service.
            project_id (str): The Project ID for a valid watsonx.ai project.
            model_name (str): The name of the model that this `WatsonXConnection` should
                connect to. See the `API model_id` column on
                [this](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx#ibm-provided)
                page for more information.
            location (str): The location of the service. E.g. `eu-gb` for Great Britain.
                See [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/regional-datactr.html?context=wx)
                for a complete list.
        """

        self.__model_name__ = model_name

        access_token_details = self.__get_access_token__(api_key)
        self.__access_token__ = access_token_details["access_token"]
        self.__expiration__ = access_token_details["expiration"]
        self.__project_id__ = project_id
        self.__location__ = location

        super().__init__()

    def __get_access_token__(self, api_key: str) -> dict[str, any]:
        """
        Use the IBM Access Token API to generate a new access token based on
        the provided API key.

        Args:
            api_key (str): The API key that should be used for generating an
                access token.

        Returns:
            A dictionary representing the response to the access token API.
            The `"access_token"` key contains the token itself and
            `"expiration"` contains time when the token expires.
        """

        response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        ).json()

        return response

    def __get_url__(self, page: str) -> str:
        """
        Get the endpoint URL for a given page.

        Args:
            page (str): The page that the URL will access. Possible pages include `generation`,
                `embeddings` and `chat`.

        Return:
            A string containing the URL with the provided `page` embedded inside it.
        """

        return f"https://{self.__location__}.ml.cloud.ibm.com/ml/v1/text/{page}?version=2024-03-14"

    def __get_info__(self):
        return "Model: " + self.__model_name__ + " (through watsonx.ai)"

    def __make_request__(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: int,
        alternative_tokens: bool = False,
    ) -> ModelResponse:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = requests.post(
            self.__get_url__("chat"),
            headers={
                "Authorization": f"Bearer {self.__access_token__}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "messages": messages,
                "model_id": self.__model_name__,
                "project_id": self.__project_id__,
                "temperature": temperature,
                "logprobs": alternative_tokens,
                "top_logprobs": 5 if alternative_tokens else 0,
                "max_tokens": 128,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tried to generate response but got errors: {response.json()['errors']}"
            )

        response_json = response.json()
        choice = response_json["choices"][0]
        model_response = ModelResponse("")

        if alternative_tokens:
            data = response_json["choices"][0]["logprobs"]["content"]
            model_response = self.__calculate_alternate_tokens__(data)

        model_response.message = choice["message"]["content"]
        model_response.generated_tokens_count = response_json["usage"][
            "completion_tokens"
        ]

        return model_response

    def __calculate_embeddings__(self, prompts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.__get_url__("embeddings"),
            headers={
                "Authorization": f"Bearer {self.__access_token__}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "inputs": prompts,
                "model_id": "ibm/granite-embedding-107m-multilingual",  # TODO: Support specific embedding model for model_name
                "project_id": self.__project_id__,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tried to generate embedding but got errors: {response.json()['errors']}"
            )

        return [r["embedding"] for r in response.json()["results"]]

    def __mediate__(
        self, prompt: str, data: list[any], format: dict[any, any], temperature: int
    ) -> Optional[dict[any, any]]:
        response = requests.post(
            self.__get_url__("chat"),
            headers={
                "Authorization": f"Bearer {self.__access_token__}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model_id": self.__model_name__,
                "project_id": self.__project_id__,
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": json.dumps({"input": data})}
                        ],
                    },
                ],
                "temperature": temperature,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tried to mediate but got code {response.status_code} and errors: {response.json()['errors']}"
            )

        try:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except json.JSONDecodeError:
            return None

    def __calculate_alternate_tokens__(self, data: list[any]) -> ModelResponse:
        """
        Rearrange alternate token data into forms that can be used for different
        visualizations.

        Args:
            data (list[any]): The raw data to be rearranged.

        Returns:
            A `ModelResponse` where all properties but the `message` property are
            filled based on the alternative token data.
        """

        model_response = ModelResponse("")

        for result in data:
            selected = Token(
                text=result["token"],
                prob=result.get("logprob", 0.0),
            )
            model_response.candidate_token_groups.append([])
            model_response.alternative_tokens.append([selected.text, []])
            selected_index = 1
            model_response.prob_sum += selected.prob

            for i, token in enumerate(
                sorted(
                    result["top_logprobs"],
                    key=lambda r: r.get("logprob", 0.0),
                    reverse=True,
                )
            ):
                tok = Token(
                    text=token["token"],
                    # logprob can sometimes not be included for some reason,
                    # so default to 0
                    prob=token.get("logprob", 0.0),
                )

                if tok.text == selected.text:
                    # This token was the selected one
                    # (add 1 for compatibility with
                    # other approaches for getting
                    # the alternative tokens)
                    selected_index = i + 1

                model_response.candidate_token_groups[-1].append(tok)
                model_response.alternative_tokens[-1][1].append([tok.text, tok.prob])

            model_response.selected_indices.append(selected_index)

            if selected_index > len(result["top_logprobs"]):
                model_response.fallback_tokens.append(selected)
                model_response.alternative_tokens[-1][1].append([tok.text, tok.prob])

        return model_response
