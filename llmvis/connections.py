import abc
import asyncio
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llmvis.core.unit_importance import Combinator
from llmvis.visualization import start_visualization
from llmvis.visualization.visualization import TextHeatmap

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
    
    def word_importance_gen_shapley(self, prompt: str, sampling_ratio: float = 0.5):
        """
        Calculate the importance of each word in a given prompt
        using the TokenSHAP algorithm (see https://arxiv.org/html/2407.10114v2)
        and display a visualization of this.

        Args:
            prompt (str): The prompt that should have its
                word importance calculated
            sampling_ratio (float): How many random samples should
                be carried out (0.0 for none)
        """

        # Start responses
        responses = [self.__make_request(prompt)]
        # Nothing fancy needed for 'tokenizing' in terms of words, only splitting by spaces
        separated_prompt = prompt.split(' ')
        combinator = Combinator(separated_prompt)

        for combination in combinator.get_combinations():
            responses.append(self.__make_request(self.__flatten_words(combination, delimiter = ' ')))

        # Calculate TF-IDF representation of each response
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(responses).toarray()
        # Use TF-IDF representation to calculate similarity between each
        # response and the full response
        similarities = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:]).flatten()
        
        # Start the visualization
        heatmap = TextHeatmap(units = separated_prompt,
                              weights = combinator.get_shapley_values(similarities))
        start_visualization(heatmap)
                
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
    def __make_request(self, prompt: str) -> str:
        """
        Make a request to this connection using a prompt.

        Args:
            prompt (str): The prompt to be given to the connected model
        
        Returns:
            The model's generated response to the prompt
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
    
    def _Connection__make_request(self, prompt: str) -> str:
        return ollama.generate(model = self._model_name, prompt = prompt)['response']