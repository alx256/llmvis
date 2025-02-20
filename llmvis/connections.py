from __future__ import annotations
import abc
import asyncio
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
from llmvis.custom_visualizations import WordSpecificLineChart, AIClassifier

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

    def word_importance_gen_shapley(
        self, prompt: str, sampling_ratio: float = 0.5
    ) -> Visualizer:
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
        responses = [self.__make_request(prompt, temperature=0.0)]
        # Nothing fancy needed for 'tokenizing' in terms of words, only splitting by spaces
        separated_prompt = prompt.split(" ")
        combinator = Combinator(separated_prompt)
        requests = [prompt]

        for i, combination in enumerate(combinator.get_combinations()):
            request = self.__flatten_words(combination, delimiter=" ")
            responses.append(self.__make_request(request, temperature=0.0))

            combination_local = combination.copy()
            combination_local.insert(i, "_" * len(separated_prompt[i]))
            formatted_request = self.__flatten_words(combination_local, delimiter=" ")

            requests.append(formatted_request)

        # Calculate TF-IDF representation of each response
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(responses).toarray()
        # Use TF-IDF representation to calculate similarity between each
        # response and the full response
        similarities = cosine_similarity(
            vectors[0].reshape(1, -1), vectors[1:]
        ).flatten()

        # Start the visualization
        shapley_vals = combinator.get_shapley_values(similarities)
        units = [
            Unit(
                separated_prompt[i],
                shapley_vals[i],
                [
                    ("Shapley Value", shapley_vals[i]),
                    ("Generated Prompt", responses[i + 1]),
                ],
            )
            for i in range(len(separated_prompt))
        ]
        table_contents = [
            [
                requests[i],
                responses[i],
                separated_prompt[i - 1] if i > 0 else "N/A",
                str(shapley_vals[i - 1]) if i > 0 else "N/A",
                str(similarities[i - 1]) if i > 0 else "N/A",
            ]
            for i in range(len(responses))
        ]

        table_heatmap = TableHeatmap(
            contents=table_contents,
            headers=[
                "Prompt",
                "Model Response",
                "Missing Term",
                "Shapley Value",
                "Cosine Difference",
            ],
            weights=[0.0] + shapley_vals,
        )
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

        separated_prompt = prompt.split(" ")
        combinator = Combinator(separated_prompt)

        requests = [prompt] + [
            self.__flatten_words(combination, delimiter=" ")
            for combination in combinator.get_combinations()
        ]
        embeddings = []

        for i, request in enumerate(requests):
            print(f"Calculating {i}/{len(requests)}...")
            embeddings.append(self.__calculate_embeddings(request)[0])

        embeddings = np.array(embeddings)

        similarities = cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1:]
        ).flatten()
        shapley_vals = combinator.get_shapley_values(similarities)
        units = [
            Unit(
                separated_prompt[i],
                shapley_vals[i],
                [("Shapley Value", shapley_vals[i]), ("Embedding", embeddings[i + 1])],
            )
            for i in range(len(separated_prompt))
        ]

        # Use PCA to reduce dimensionality to suppress noise and speed up
        # computations (per scikit-learn recommendations
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        text_heatmap = TextHeatmap(units)
        text_heatmap.set_comments(self.__get_info__())
        tag_cloud = TagCloud(units)
        tag_cloud.set_comments(self.__get_info__())
        scatter_plot = ScatterPlot(reduced)
        scatter_plot.set_comments(self.__get_info__())

        return Visualizer([text_heatmap, tag_cloud, scatter_plot])

    def k_temperature_sampling(
        self,
        prompt: str,
        k: int,
        start: float = 0.0,
        end: float = 1.0,
        enable_mediation: bool = False,
        mediator_connection: Optional[Connection] = None,
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
            start (float): The starting temperature value. Default
                is 0.0. Must be less than `end`.
            end (float): The ending temperature value. Default is
                1.0. Must be greater than `start`.
            enable_mediation (bool): Set to `True` to enable the
                AI mediator tools. These use an external "mediator"
                model to visualize additional insights about results.
                Enabling this will show an additional visualization
                containing the different overlapping concepts for
                each temperature sample and a line chart with the
                detected hallucinations for each sample.
                Default is `False`. Note that enabling this will take
                additional computation time and has the potential to be
                inaccurate.
            mediator_connection (Optional[Connection]): The `Connection`
                that should be used for mediator visualizations.
                Setting this to `None` will just use this `Connection`.
                Default is `None`.

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
            sample = self.__make_request(prompt=prompt, temperature=t)

            samples.append(sample)

            table_contents.append([t, sample])
            temperatures.append(t)

            for word in sample.split(" "):
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
            contents=table_contents, headers=["Sampled Temperature", "Sample Result"]
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
        bar_chart = BarChart(frequencies_list)
        bar_chart.set_comments(self.__get_info__())

        line_chart = WordSpecificLineChart(temperature_change_frequencies, t_values)
        line_chart.set_comments(self.__get_info__())

        visualizations = [table_heatmap, bar_chart, line_chart]

        if enable_mediation:
            response_data = None
            attempts = 0
            t = 0.0

            while response_data is None:
                if attempts >= MEDIATION_ATTEMPTS:
                    break

                response_data = (mediator_connection or self).__mediate__(
                    prompt=MEDIATION_PROMPT,
                    data=table_contents,
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

    def __get_info__(self):
        return "Model: " + self._model_name + " (through Ollama)"

    def _Connection__make_request(self, prompt: str, temperature: int) -> str:
        return ollama.generate(
            model=self._model_name, prompt=prompt, options={"temperature": temperature}
        ).response

    def _Connection__calculate_embeddings(
        self, prompts: list[str]
    ) -> list[list[float]]:
        return ollama.embed(model=self._model_name, input=prompts).embeddings

    def __mediate__(
        self, prompt: str, data: list[any], format: dict[any, any], temperature: int
    ) -> Optional[dict[any, any]]:
        try:
            response = ollama.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"input": data})},
                ],
                options={"temperature": temperature},
                format=format,
            ).message.content
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

    def __init__(self, api_key: str, project_id: str, model_name: str):
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
        """

        self.__model_name__ = model_name

        access_token_details = self.__get_access_token__(api_key)
        self.__access_token__ = access_token_details["access_token"]
        self.__expiration__ = access_token_details["expiration"]
        self.__project_id__ = project_id

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

        return f"https://eu-gb.ml.cloud.ibm.com/ml/v1/text/{page}?version=2023-05-29"

    def __get_info__(self):
        return "Model: " + self.__model_name__ + " (through watsonx.ai)"

    def _Connection__make_request(self, prompt: str, temperature: int) -> str:
        response = requests.post(
            self.__get_url__("generation"),
            headers={
                "Authorization": f"Bearer {self.__access_token__}",
                "Content-Type": "application/json",
            },
            json={
                "model_id": self.__model_name__,
                "input": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "time_limit": 1000,
                    "temperature": temperature,
                },
                "project_id": self.__project_id__,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tried to generate response but got errors: {response.json()['errors']}"
            )

        return " ".join([c["generated_text"] for c in response.json()["results"]])

    def _Connection__calculate_embeddings(self, prompts: list[str]):
        response = requests.post(
            self.__get_url__("embeddings"),
            headers={
                "Authorization": f"Bearer {self.__access_token__}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "inputs": prompts,
                "model_id": "granite-embedding-107m-multilingual",  # TODO: Support specific embedding model for model_name
                "project_id": self.__project_id__,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tried to generate embedding but got errors: {response.json()['errors']}"
            )

        return [str(r) for r in response.json()["results"]["embedding"]]

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
