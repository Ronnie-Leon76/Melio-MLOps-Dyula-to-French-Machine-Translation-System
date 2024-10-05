"""
Main module for a machine translation inference server
"""
import re
import argparse
from typing import List
import torch


try:
    from joeynmt.prediction import predict, prepare
    from joeynmt.config import load_config, parse_global_args
    from kserve.utils.utils import generate_uuid
    from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
except ImportError as e:
    print(f"Error importing required modules: {e}")
    raise


MODEL_CONFIG_PATH = "./models/config.yaml"
CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]]'


def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()


class JoeyNMTModelDyuFr:
    """
    JoeyNMTModelDyuFr which load JoeyNMT model for inference.
    :param config_path: Path to YAML config file
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    """
    def __init__(self, config_path: str, n_best: int = 1) -> None:
        seed = 42
        torch.manual_seed(seed)
        cfg = load_config(config_path)
        model_args = parse_global_args(cfg, rank=0, mode="translate")
        self.args = model_args._replace(test=model_args.test._replace(n_best=n_best))
        # build model
        self.model, _, _, self.test_data = prepare(self.args, rank=0, mode="translate")

    def _translate_data(self) -> List[str]:
        _, _, hypotheses, _, _, _ = predict(
            model=self.model,
            data=self.test_data,
            compute_loss=False,
            device=self.args.device,
            rank=0,
            n_gpu=self.args.n_gpu,
            normalization="none",
            num_workers=self.args.num_workers,
            args=self.args.test,
            autocast=self.args.autocast,
        )
        return hypotheses

    def translate(self, sentence) -> List[str]:
        """
        Translate the given sentence.

        :param sentence: Sentence to be translated
        :return:
        - translations: (list of str) possible translations of the sentence.
        """
        self.test_data.set_item(sentence.strip())
        translations = self._translate_data()
        assert len(translations) == len(self.test_data) * self.args.test.n_best
        self.test_data.reset_cache()
        return translations

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        """
        return {
            "config_path": MODEL_CONFIG_PATH,
            "n_best": self.args.test.n_best,
            "device": str(self.args.device),
        }


class MyModel(Model):
    """
    MyModel class that loads and handles the translation model.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        """
        Load the JoeyNMT model.
        """
        self.model = JoeyNMTModelDyuFr(config_path=MODEL_CONFIG_PATH, n_best=1)
        self.ready = True

    @staticmethod
    def preprocess(payload: InferRequest) -> List[str]:
        """
        Preprocess the payload by cleaning the input text.
        :param payload: Inference request payload
        :return: List of cleaned text strings
        """
        infer_inputs: List[str] = payload.inputs[0].data
        cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        return cleaned_texts

    def predict(self, data: List[str]) -> InferResponse:
        """
        Generate predictions from the model.
        :param data: List of preprocessed text strings
        :return: Inference response containing the predictions
        """
        response_id = generate_uuid()
        results: List[str] = [self.model.translate(sentence=s)[0] for s in data]
        #print(f"** result ({type(results)}): {results}")

        infer_output = InferOutput(
            name="output-0", shape=[len(results)], datatype="STR", data=results
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name",
    default="model",
    help="The name that the model is served under."
)
parsed_args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(parsed_args.model_name)
    ModelServer().start([model])
