"""Causal language model-based prompt difficulty classifier.

Implements a fine-tuned LLM that predicts prompt difficulty scores
for routing decisions.
"""

import re
import time
from typing import List

import numpy as np
import torch

from routellm.routers.causal_llm.configs import ModelTypeEnum, RouterModelConfig
from routellm.routers.causal_llm.llm_utils import get_model, get_tokenizer
from routellm.routers.causal_llm.prompt_format import PromptFormat


class CausalLLMClassifier:
    """Fine-tuned LLM for predicting prompt difficulty scores.

    Uses special tokens [[1]] through [[5]] to indicate difficulty level.
    Higher scores indicate harder prompts requiring strong model.
    """
    def __init__(
        self,
        config: RouterModelConfig,
        ckpt_local_path: str,
        prompt_format: PromptFormat,
        score_threshold: int,
        prompt_field: str = "messages",
        use_last_turn: bool = False,
        additional_fields: List[str] = list(["label", "pidx"]),
        max_new_tokens: int = 6,
    ):
        """Initialize causal LLM classifier.

        Predicts prompt difficulty on scale [1, 5]. Higher scores indicate
        stronger model is needed. Routing probability is computed as:
        P(route_to_strong) = sum(prob(score) for score >= score_threshold).

        Parameters
        ----------
        config : RouterModelConfig
            Model configuration.
        ckpt_local_path : str
            Path to fine-tuned model checkpoint.
        prompt_format : PromptFormat
            Prompt formatting template.
        score_threshold : int
            Score threshold for routing decision.
        prompt_field : str, optional
            Field name for prompts in input (default "messages").
        use_last_turn : bool, optional
            Whether to use only last turn (default False).
        additional_fields : list[str], optional
            Additional fields to include (default ["label", "pidx"]).
        max_new_tokens : int, optional
            Maximum tokens to generate (default 6).
        """
        # Initialize the batch generator
        print(f"Loading model checkpoint from {ckpt_local_path} ...")
        s = time.time()

        assert config.model_type == ModelTypeEnum.CAUSAL

        # assert that config has 5 special tokens with the format [[rating]]
        assert len(config.special_tokens) == config.num_outputs
        for i in range(1, config.num_outputs + 1):
            assert f"[[{i}]]" in config.special_tokens

        model = get_model(config=config, model_ckpt=ckpt_local_path)
        self.model = model.to("cuda").eval()

        self.prompt_format = prompt_format
        self.use_last_turn = use_last_turn
        self.prompt_field = prompt_field
        self.additinal_fields = additional_fields

        self.tokenizer = get_tokenizer(
            config.model_id,
            special_tokens=config.special_tokens,
            truncation_side="left",
            padding_side="left",
        )
        self.orig_vocab_size = len(self.tokenizer) - config.num_outputs
        self.max_new_tokens = max_new_tokens
        self.score_threshold = score_threshold
        assert (
            self.score_threshold == config.num_outputs - 1
        ), "this is the default value for now."
        print(f"Done loading model in {time.time() - s} seconds.")

    def preprocess(self, row):
        """Prepare prompt before feeding to the model.

        Extracts messages from the specified field, formats them according
        to the prompt template, tokenizes, and encodes into input IDs.

        Parameters
        ----------
        row : dict
            Input row containing prompt data with keys matching prompt_field
            and additional_fields.

        Returns
        -------
        dict
            Preprocessed data with 'input_ids' and additional fields.
        """
        # add additional fields to the final output (e.g. for later evaluation)
        data_row = {}
        for field in self.additinal_fields:
            data_row[field] = row[field]
        # select turns from the prompt field
        openai_messages = (
            row[self.prompt_field]
            if self.use_last_turn
            else row[self.prompt_field][:-1]
        )
        # convert openai messages formot to model's prompt format
        text = self.prompt_format.generate_prompt(openai_messages)
        # tokenize and encode
        data_row["input_ids"] = np.array(self.tokenizer.encode(text))

        return data_row

    def __call__(self, row):
        row = self.preprocess(row)
        input_ids = torch.as_tensor(row["input_ids"]).to("cuda").reshape(1, -1)
        with torch.no_grad():
            output_new = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # see https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L101
        row["output_ids"] = output_new.sequences.squeeze()[input_ids.shape[1] :].cpu()
        assert len(row["output_ids"]) == len(row["output_ids"])

        # find the first token within the special tokens range. This is our score prediction.
        label_token_idx = next(
            (i for i, x in enumerate(row["output_ids"]) if x >= self.orig_vocab_size),
            None,
        )
        if label_token_idx is None:
            return None

        # extract logits of predicted labels from scores of each output token
        # (check hf github modeling llama)
        score_logits = np.array(
            output_new.scores[label_token_idx][0].to("cpu")[self.orig_vocab_size :]
        )
        row["score_logits"] = score_logits
        binary_prob, softmax_scores = self.compute_routing_prob(score_logits)
        row["softmax_scores"] = softmax_scores
        row["binary_prob"] = binary_prob

        row = self.postprocess(row)
        return row

    def compute_routing_prob(self, score_logits):
        """Convert score logits to binary routing probability.

        Applies softmax to score logits and sums probabilities of scores
        at or above the routing threshold to produce binary probability.

        Parameters
        ----------
        score_logits : np.ndarray
            Raw logit scores from the model for each difficulty level.

        Returns
        -------
        tuple
            (binary_prob, softmax_scores) where binary_prob is the
            probability of routing to strong model and softmax_scores
            are the normalized probabilities for each difficulty level.
        """
        exp_scores = np.exp(score_logits - np.max(score_logits))
        softmax_scores = exp_scores / np.sum(exp_scores)
        binary_prob = np.sum(softmax_scores[self.score_threshold - 1 :])
        return binary_prob, softmax_scores

    def postprocess(self, row):
        """Post-process model predictions into readable format.

        Decodes output tokens to strings, parses score from output,
        and validates consistency between logits and generation predictions.

        Parameters
        ----------
        row : dict
            Row containing model output IDs and score logits.

        Returns
        -------
        dict
            Row with decoded output_str, output_tokens, and parsed score_pred.

        Raises
        ------
        AssertionError
            If logits prediction does not match generated score prediction.
        """

        output_str = self.tokenizer.decode(row["output_ids"])
        row["output_tokens"] = self.tokenizer.convert_ids_to_tokens(row["output_ids"])
        row["output_str"] = output_str
        row["score_pred"] = self.parse_score(output_str)
        # to debug, check both logits and generation prediction match
        logits_pred = np.argmax(row["score_logits"]) + 1
        assert logits_pred == row["score_pred"]

        # clean up input data
        del row["input_ids"]

        return row

    def parse_score(self, text):
        """Extract integer score from model output string.

        Parses text for pattern [[N]] where N is a float-formatted integer
        (1-5) indicating difficulty level prediction.

        Parameters
        ----------
        text : str
            Decoded model output string.

        Returns
        -------
        int
            Parsed difficulty score.

        Raises
        ------
        Exception
            If text does not contain valid [[N]] pattern.
        """
        match = re.search(r"\[\[([\d\.]+)\]\]", text)
        if match:
            return int(float(match.group(1)))
        else:
            raise Exception(f"Bad score format {text}.")
