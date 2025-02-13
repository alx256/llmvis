from __future__ import annotations
import abc
import asyncio
import ollama
import numpy as np
from typing_extensions import Optional
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from llmvis.core.unit_importance import Combinator
from llmvis.visualization import Visualizer
from llmvis.visualization.visualization import Unit, TextHeatmap, TableHeatmap, TagCloud, ScatterPlot, BarChart
from llmvis.core.preprocess import should_include
from llmvis.custom_visualizations import WordSpecificLineChart, AIClassifier

CLASSIFICATION_PROMPT = """You will be given some JSON data. You must first establish a number of classes for the data that broadly groups the text items together. Try to find similar concepts and ideas to establish as few classes as possible.  Then, using these classes you have created, classify each numerical item into one of the m classes.

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
\"result\":[
[\"Statements about the weather\", [0.1, 0.3, 0.4]],
[\"Statements about the time\", [0.2]]
]
}
Input: {
\"input\": [
[0.843924, \"Hamburgers are made with beef\"],
[0.3192381293, \"Football has different meanings in American and British English\"]
[0.289178237, \"Vegetables are good for you!\"],
[0.9393939, \"Coca-Cola is an American brand\"],
[0.42938293, \"Water is a hydrating beverage\"],
[0.9923231, \"Swimming is great for fitness\"],
[0.232819912, \"Chicken is rich in protein\"]
]
}
Output: {
\"result\":[
[\"Food\", [0.843924, 0.289178237, 0.232819912]],
[\"Drink\", [0.9393939, 0.42938293]],
[\"Sports\", [0.3192381293, 0.9923231]]
]
}"""

CLASSIFICATION_ATTEMPTS = 5

class Connection(abc.ABC):
    """
    Base class for connections. A 'connection' is a link between the program
    written by the user and the service that is used for hosting the LLMs.
    Override this class to add support for LLMVis integrating with another
    service.
    """

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
    
    def word_importance_gen_shapley(self, prompt: str, sampling_ratio: float = 0.5) -> Visualizer:
        """
        Calculate the importance of each word in a given prompt
        using the TokenSHAP algorithm (see https://arxiv.org/html/2407.10114v2)
        and get a `Visualizer` visualizing this of this.

        Args:
            prompt (str): The prompt that should have its
                word importance calculated
            sampling_ratio (float): How many random samples should
                be carried out (0.0 for none)

        Returns:
            A `Visualizer` showing a table heatmap, a text heatmap and a tag cloud
            for the importance of each word.
        """

        # Start responses
        responses = [self.__make_request(prompt, temperature = 0.0)]
        # Nothing fancy needed for 'tokenizing' in terms of words, only splitting by spaces
        separated_prompt = prompt.split(' ')
        combinator = Combinator(separated_prompt)
        requests = [prompt]

        for i, combination in enumerate(combinator.get_combinations()):
            request = self.__flatten_words(combination, delimiter = ' ')
            responses.append(self.__make_request(request, temperature = 0.0))

            combination_local = combination.copy()
            combination_local.insert(i, '_' * len(separated_prompt[i]))
            formatted_request = self.__flatten_words(combination_local, delimiter = ' ')

            requests.append(formatted_request)

        # Calculate TF-IDF representation of each response
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(responses).toarray()
        # Use TF-IDF representation to calculate similarity between each
        # response and the full response
        similarities = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:]).flatten()
        
        # Start the visualization
        shapley_vals = combinator.get_shapley_values(similarities)
        units = [Unit(separated_prompt[i], shapley_vals[i],
                      [('Shapley Value', shapley_vals[i]),
                       ('Generated Prompt', responses[i + 1])])
                    for i in range(len(separated_prompt))]
        table_contents = [[requests[i], responses[i],
            separated_prompt[i - 1] if i > 0 else 'N/A',
            str(shapley_vals[i - 1]) if i > 0 else 'N/A',
            str(similarities[i - 1]) if i > 0 else 'N/A'] for i in range(len(responses))]

        table_heatmap = TableHeatmap(contents = table_contents,
                                     headers = ['Prompt', 'Model Response', 'Missing Term', 'Shapley Value', 'Cosine Difference'],
                                     weights = [0.0] + shapley_vals)
        table_heatmap.set_comments(self.__get_info__())
        text_heatmap = TextHeatmap(units)
        text_heatmap.set_comments(self.__get_info__())
        tag_cloud = TagCloud(units)
        tag_cloud.set_comments(self.__get_info__())

        return Visualizer([table_heatmap, text_heatmap, tag_cloud])

    def word_importance_embed_shapley(self, prompt: str) -> Visualizer:
        """
        Calculate the importance of each word in a given prompt
        using embeddings to approximate the importance of each
        word in the sentence and get a `Visualizer` visualizing
        this.

        Args:
            prompt (str): The prompt that should have its word
                importance calculated

        Returns:
            A `Visualizer` showing a text heatmap and tag cloud
            for the importance of each word.
        """

        separated_prompt = prompt.split(' ')
        combinator = Combinator(separated_prompt)

        requests = [prompt] + [self.__flatten_words(combination, delimiter = ' ')
                                for combination in combinator.get_combinations()]
        embeddings = []

        for i, request in enumerate(requests):
            print(f'Calculating {i}/{len(requests)}...')
            embeddings.append(self.__calculate_embeddings(request)[0])

        embeddings = np.array(embeddings)

        similarities = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1:]).flatten()
        shapley_vals = combinator.get_shapley_values(similarities)
        units = [Unit(separated_prompt[i], shapley_vals[i],
                        [('Shapley Value', shapley_vals[i]),
                         ('Embedding', embeddings[i + 1])])
                    for i in range(len(separated_prompt))]

        # Use PCA to reduce dimensionality to suppress noise and speed up
        # computations (per scikit-learn recommendations
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
        pca = PCA(n_components = 2)
        reduced = pca.fit_transform(embeddings)

        text_heatmap = TextHeatmap(units)
        text_heatmap.set_comments(self.__get_info__())
        tag_cloud = TagCloud(units)
        tag_cloud.set_comments(self.__get_info__())
        scatter_plot = ScatterPlot(reduced)
        scatter_plot.set_comments(self.__get_info__())

        return Visualizer([text_heatmap, tag_cloud, scatter_plot])

    def k_temperature_sampling(self, prompt: str, k: int,
                               start: float = 0.0, end: float = 1.0,
                               use_ai_classifier: bool = False,
                               ai_classifier_connection: Optional[Connection] = None) -> Visualizer:
        """
        Sample `k` temperature values starting with `start` and ending
        with `end` to examine the differences between temperature
        values and get a `Visualizer` visualizing this.

        Args:
            prompt (str): The prompt that should be used with the
                different temperature values.
            k (int): The number of samples that should be used. Must
                be greater than 2.
            start (float): The starting temperature value. Default
                is 0.0. Must be less than `end`.
            end (float): The ending temperature value. Default is
                1.0. Must be greater than `start`.
            use_ai_classifier (bool): Set to `True` to enable the AI
                classifier visualization which will show an
                AI-generated overview of the different concepts for
                each temperature sample and `False` to disable this.
                Default is `False`. Note that enabling this will take
                additional computation time compared to other visualizations
                and has the potential to be inaccurate.
            ai_classifier_connections (Optional[Connection]): The `Connection`
                that should be used for the AI classifier visualization.
                Setting this to `None` will just use this `Connection`.
                Default is `None`.

        Returns:
            A `Visualizer` showing a table containing the results of the
            different samples.
        """

        if k < 2:
            raise RuntimeError('k must be at least 2.')

        if end < start:
            raise RuntimeError('start must come before the end.')

        if end == start:
            raise RuntimeError('start and end cannot be the same.')

        step = (start + end) / (k - 1)
        t = start
        # Sample results
        samples = []
        # Temperature values
        temperatures = []
        # Store the frequency of each word to visualize as a bar chart
        frequencies = {}
        # Store the frequencies as temperatures change to visualize as a line chart
        temperature_change_frequencies = {}

        table_contents = []
        t_values = []

        for _ in range(k):
            t_values.append(t)
            sample = self.__make_request(prompt = prompt, temperature = t)

            samples.append(sample)

            table_contents.append([t, sample])
            temperatures.append(t)

            for word in sample.split(' '):
                # Only keep alpha-numeric (i.e. ignore punctuation) chars
                # of the string
                chars = ''.join(e.lower() for e in word if e.isalnum())

                # Edge case where a "word" might contain no alphanumeric characters
                # for example just a dash '-' surrounded by spaces
                if len(chars) == 0:
                    continue

                frequencies.setdefault(chars, 0)
                frequencies[chars] += 1
                temperature_change_frequencies.setdefault(chars, [])

                # Calculate the frequencies for each temperature value for this word
                if len(temperature_change_frequencies[chars]) > 0 and temperature_change_frequencies[chars][-1][0] == t:
                    temperature_change_frequencies[chars][-1][1] += 1
                else:
                    temperature_change_frequencies[chars].append([t, 1])

            t += step

        table_heatmap = TableHeatmap(contents = table_contents,
                                     headers = ['Sampled Temperature', 'Sample Result'])
        table_heatmap.set_comments(self.__get_info__())

        # Final frequencies list in expected format for bar chart 
        frequencies_list = []

        for name in frequencies.keys():
            # Only keep interesting words
            if should_include(name):
                frequencies_list.append([name, frequencies[name]])

        # Sort in ascending order
        frequencies_list = sorted(frequencies_list,
                                  key = lambda entry : entry[1],
                                  reverse = True)
        # Don't want too many bars on the bar chart
        frequencies_list = frequencies_list[:7]
        bar_chart = BarChart(frequencies_list)
        bar_chart.set_comments(self.__get_info__())

        line_chart = WordSpecificLineChart(temperature_change_frequencies, t_values)
        line_chart.set_comments(self.__get_info__())

        visualizations = [table_heatmap, bar_chart, line_chart]

        if use_ai_classifier:
            classified_data = None
            attempts = 0
            t = 0.0

            while classified_data is None:
                if attempts >= CLASSIFICATION_ATTEMPTS:
                    break

                classified_data = (ai_classifier_connection or self).__classify(table_contents, temperature = t)
                attempts += 1
                t += 1.0 / CLASSIFICATION_ATTEMPTS

            ai_classifier = AIClassifier(classified_data, temperatures)
            ai_classifier.set_comments(self.__get_info__())
            visualizations.append(ai_classifier)

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

        words_str = ''
        for i, word in enumerate(words):
            words_str += word

            # Add a separator between each word and the next
            if i < len(words) - 1:
                words_str += delimiter
        
        return words_str

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
    def __make_request(self, prompt: str, temperature: int) -> str:
        """
        Make a request to this connection using a prompt.

        Args:
            prompt (str): The prompt to be given to the connected model
            temperature (int): The temperature that should be used for
                generation. 0.0 means deterministic behaviour while higher
                temperatures introduce more nondeterminism.
        
        Returns:
            The model's generated response to the prompt
        """
        pass

    @abc.abstractmethod
    def __calculate_embeddings(self, prompts: list[str]) -> list[list[float]]:
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
    def __classify(self, data: list[list[any]], temperature: int) -> list[list[any]]:
        """
        Classify some data into some number of classes.

        Args:
            data (list[list[any]]): The data that should be classified. Each
                element of this list should be another list where the first
                element is the numerical value that will be put into a class
                and the second element is a string that will be used for
                determining the class.
        
        Returns:
            A list where each element is another list with the first element being
            a string containing a class name and the second element is a list of
            all numerical values from the input data that are determined to belong
            to this class.
        """

        pass

class OllamaConnection(Connection):
    _model_name = ""

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
        if ollama.pull(model_name)['status'] != 'success':
            raise RuntimeError(model_name + ' was not able to be pulled. Check that it is a supported model.')
    
    def __get_info__(self):
        return 'Model: ' + self._model_name + ' (through Ollama)'

    def _Connection__make_request(self, prompt: str, temperature: int) -> str:
        return ollama.generate(model = self._model_name,
                               prompt = prompt,
                               options = {
                                   'temperature' : temperature
                               }).response

    def _Connection__calculate_embeddings(self, prompts: list[str]) -> list[list[float]]:
        return ollama.embed(model = self._model_name, input = prompts).embeddings
    
    def _Connection__classify(self, data: list[list[any]], temperature: int) -> Optional[list[list[any]]]: 
        try:
            response = ollama.chat(model = self._model_name,
                                                messages = [
                                                    {'role': 'system', 'content': CLASSIFICATION_PROMPT},
                                                    {'role': 'user', 'content': json.dumps({'input': data})}],
                                                options = {
                                                    'temperature': temperature
                                                }
                ).message.content
            return json.loads(response)['result']
        except json.decoder.JSONDecodeError:
            # LLM gave a response that was unexpected as it did not only contain
            # JSON data.

            return None