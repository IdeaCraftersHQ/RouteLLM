"""Abstract base classes and implementations for prompt routing.

This module provides router classes that compute confidence scores for routing
prompts to either strong or weak models based on difficulty estimation.
"""

import abc
import functools
import random

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)


def no_parallel(cls):
    """Mark router class as non-parallelizable.

    Some routers (e.g., those with randomness or state mutation) cannot be
    safely parallelized. This decorator marks them.

    Parameters
    ----------
    cls : type
        Router class to mark.

    Returns
    -------
    type
        The decorated class.
    """
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    """Abstract base class for prompt routers.

    Routers compute a confidence score (0-1) indicating the likelihood that
    a prompt should be routed to the strong model. A threshold is applied
    to make the routing decision: if score >= threshold, route to strong;
    otherwise route to weak.
    """
    NO_PARALLEL = False

    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        """Calculate confidence score for routing to strong model.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Confidence score in [0, 1] representing estimated win rate of the
            strong model. Score >= threshold routes to strong; score < threshold
            routes to weak.
        """
        pass

    def route(self, prompt, threshold, routed_pair):
        """Route prompt to strong or weak model based on threshold.

        Parameters
        ----------
        prompt : str
            Input prompt to route.
        threshold : float
            Decision threshold in [0, 1]. If calculate_strong_win_rate >=
            threshold, route to strong; otherwise route to weak.
        routed_pair : ModelPair
            Pair of strong and weak model names.

        Returns
        -------
        str
            Name of model to route to (either routed_pair.strong or
            routed_pair.weak).
        """
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        """Return router class name as string."""
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    """Route prompts using fine-tuned causal language model.

    This router uses a causal LLM (e.g., Llama-3-8B) fine-tuned to predict
    prompt difficulty. Scores are obtained via probability of special tokens
    indicating difficulty levels.

    Non-parallelizable: maintains model state during inference.
    """

    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        """Initialize causal LLM router.

        Parameters
        ----------
        checkpoint_path : str
            HuggingFace model ID or path to fine-tuned router model.
        score_threshold : float, optional
            Logit threshold for binary classification (default 4).
        special_tokens : list[str], optional
            Special tokens for difficulty levels (default 5 levels).
        num_outputs : int, optional
            Number of output classes (default 5).
        model_type : str, optional
            Model architecture type (default "causal").
        model_id : str, optional
            Base model ID from HuggingFace (default "meta-llama/Meta-Llama-3-8B").
        flash_attention_2 : bool, optional
            Enable Flash Attention 2 optimization (default False).
        """
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        """Calculate strong model win rate using causal LLM score.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Strong model win rate in [0, 1]. Returns 1.0 if output is invalid
            (routes to strong model as fallback).
        """
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    """Route prompts using BERT-based sequence classifier.

    This router uses a BERT model fine-tuned for prompt difficulty
    classification. Converts multi-class probabilities to binary routing
    scores by summing probabilities of "hard" classes.

    Non-parallelizable: maintains model state during inference.
    """

    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        """Initialize BERT router.

        Parameters
        ----------
        checkpoint_path : str
            HuggingFace model ID or path to fine-tuned BERT router.
        num_labels : int, optional
            Number of classification labels (default 3).
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
        """Calculate strong model win rate using BERT classification.

        Sums softmax probabilities of hard classes (last 2 of 3) and returns
        1 - (sum of hard probs) as the strong model win rate.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Strong model win rate in [0, 1].
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    """Route prompts using Similarity Weighted ranking algorithm.

    Finds similar conversations in arena battle dataset using embeddings,
    then computes Elo-based win rates weighted by similarity. Routes based
    on expected strong model win rate relative to weak model.
    """

    def __init__(
        self,
        arena_battle_datasets,
        arena_embedding_datasets,
        # This is the model pair for Elo calculations at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        num_tiers=10,
    ):
        """Initialize similarity weighted ranking router.

        Parameters
        ----------
        arena_battle_datasets : list[str]
            HuggingFace dataset IDs containing model battle comparisons.
        arena_embedding_datasets : list[str]
            HuggingFace dataset IDs containing conversation embeddings
            corresponding to arena_battle_datasets.
        strong_model : str, optional
            Model name for strong model in Elo calculations (default
            "gpt-4-1106-preview").
        weak_model : str, optional
            Model name for weak model in Elo calculations (default
            "mixtral-8x7b-instruct-v0.1").
        num_tiers : int, optional
            Number of tiers for model grouping (default 10).
        """
        self.strong_model = strong_model
        self.weak_model = weak_model

        self.arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        self.arena_df = preprocess_battles(self.arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        self.arena_conv_embedding = np.concatenate(embeddings)
        self.embedding_model = "text-embedding-3-small"

        assert len(self.arena_df) == len(
            self.arena_conv_embedding
        ), "Number of battle embeddings is mismatched to data"

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    def get_weightings(self, similarities):
        """Compute exponential weightings from similarity scores.

        Parameters
        ----------
        similarities : np.ndarray
            Array of cosine similarities in [-1, 1].

        Returns
        -------
        np.ndarray
            Exponentially scaled weights favoring high-similarity samples.
        """
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        """Calculate strong model win rate using weighted Elo estimates.

        Embeds prompt, finds similar conversations, computes weighted Elo
        ratings, and returns expected win rate.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Strong model expected win rate in [0, 1].
        """
        prompt_emb = (
            (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )
        similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
            np.linalg.norm(self.arena_conv_embedding, axis=1)
            * np.linalg.norm(prompt_emb)
        )

        weightings = self.get_weightings(similarities)
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    """Route prompts using matrix factorization model.

    Factorizes model-text interactions into latent embeddings, then predicts
    win rates for model pairs given prompts.

    Non-parallelizable: maintains model state during inference.
    """

    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        """Initialize matrix factorization router.

        Parameters
        ----------
        checkpoint_path : str
            HuggingFace model ID or path to fine-tuned MF router.
        strong_model : str, optional
            Strong model name for win rate prediction (default
            "gpt-4-1106-preview").
        weak_model : str, optional
            Weak model name for win rate prediction (default
            "mixtral-8x7b-instruct-v0.1").
        hidden_size : int, optional
            Latent embedding dimension (default 128).
        num_models : int, optional
            Number of models in the factorization (default 64).
        text_dim : int, optional
            Text embedding dimension (default 1536).
        num_classes : int, optional
            Number of output classes (default 1).
        use_proj : bool, optional
            Whether to use projection layer (default True).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        """Calculate strong model win rate using MF prediction.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Strong model win rate in [0, 1].
        """
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    """Route prompts using uniform random score.

    Returns a uniformly random score in [0, 1] for each prompt. Used for
    testing and baseline comparisons.

    Non-parallelizable: randomness is non-deterministic under parallelism.
    """

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        """Calculate strong model win rate as random uniform value.

        Parameters
        ----------
        prompt : str
            Input prompt (unused).

        Returns
        -------
        float
            Random score uniformly sampled from [0, 1].
        """
        res = random.uniform(0, 1)
        print(f"[RANDOM_ROUTER] calculated win_rate={res}")
        return res


ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
