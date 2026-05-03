"""Prompt formatting utilities for causal language models.

Handles conversion of OpenAI-style message dictionaries to formatted prompts
compatible with various causal LLM templates.
"""

import copy
from typing import Dict, List

from pydantic import BaseModel, validator


class PromptFormat(BaseModel):
    """Format template for converting messages to model-specific prompts.

    Attributes
    ----------
    system : str
        System message template with {instruction} placeholder.
    assistant : str
        Assistant message template with {instruction} placeholder.
    trailing_assistant : str
        Trailing assistant message appended to prompt (usually empty).
    user : str
        User message template with {instruction} placeholder.
    default_system_message : str, optional
        Default system message if none provided (default "").
    system_in_user : bool, optional
        Whether to embed system message within user message (default False).
    is_generation : bool, optional
        Whether format is for generation (vs training/eval) (default False).
    """
    system: str
    assistant: str
    trailing_assistant: str
    user: str

    default_system_message: str = ""
    system_in_user: bool = False
    is_generation: bool = False

    @validator("system")
    def check_system(cls, value):
        assert value and (
            "{instruction}" in value
        ), "system must be a string containing '{instruction}'"
        return value

    @validator("assistant")
    def check_assistant(cls, value):
        assert (
            value and "{instruction}" in value
        ), "assistant must be a string containing '{instruction}'"
        return value

    @validator("user")
    def check_user(cls, value):
        assert value and (
            "{instruction}" in value
        ), "user must be a string containing '{instruction}'"
        return value

    @validator("system_in_user")
    def check_system_in_user(cls, value):
        # `system_in_user` is restricted to be True.
        # Re-evaluate the code and add unit tests when relaxing this.
        # assert value
        return value

    @validator("default_system_message")
    def check_default_system_message(cls, value):
        # User should explicitly give a system message if so preferred.
        # assert value == ""
        return value

    @validator("trailing_assistant")
    def check_trailing_assistant(cls, value):
        # `trailing_assistant` is restricted to be "".
        # Re-evaluate the code and add unit tests when relaxing this.
        assert value == ""
        return value

    def generate_prompt_turns(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style messages to formatted prompt turns.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list with 'role' and 'content' keys.
            Roles: 'system', 'user', 'assistant'.

        Returns
        -------
        list[dict]
            List of formatted messages with 'role' and 'content' keys.

        Raises
        ------
        ValueError
            If messages don't follow alternating user/assistant pattern or
            if last message is not from assistant (when not in generation mode).
        """
        messages = copy.deepcopy(messages)
        system_message = None
        if messages[0]["role"] == "system":
            system_message = messages.pop(0)
            if system_message["content"] is None:
                system_message["content"] = ""

        if not system_message and self.default_system_message:
            system_message = {"role": "system", "content": self.default_system_message}

        if not all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ):
            raise ValueError(
                "only supports 'system','user' and 'assistant' roles, starting with user and alternating (u/a/u/a/u...)"
            )

        if any([msg["content"] is None for msg in messages]):
            raise ValueError("Both user and assistant messages cannot be None.")

        # only applies at train/eval but not in generation
        if not self.is_generation and not messages[-1]["role"] == "assistant":
            raise ValueError(
                f"Last message must be from assistant, got {messages[-1]['role']}"
            )

        if (
            system_message is not None
            and system_message["content"]
            and not self.system_in_user
        ):
            messages.insert(0, system_message)

        prompt = []
        for message in messages:
            message_content = message["content"]
            message_content = message_content.strip()
            if message["role"] == "system":
                prompt.append(
                    {
                        "role": "system",
                        "content": self.system.format(instruction=message_content),
                    }
                )
            elif message["role"] == "user":
                if self.system_in_user:
                    prompt.append(
                        {
                            "role": "user",
                            "content": self.user.format(
                                instruction=message_content,
                                system=(
                                    self.system.format(
                                        instruction=system_message["content"]
                                    )
                                    if system_message
                                    else ""
                                ),
                            ),
                        }
                    )
                    system_message = None
                else:
                    prompt.append(
                        {
                            "role": "user",
                            "content": self.user.format(instruction=message_content),
                        }
                    )
            elif message["role"] == "assistant":
                prompt.append(
                    {
                        "role": "assistant",
                        "content": self.assistant.format(instruction=message_content),
                    }
                )

        if self.trailing_assistant:
            prompt.append({"role": "assistant", "content": self.trailing_assistant})
        return prompt

    def generate_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to single formatted prompt string.

        Applies formatting and concatenates all message contents into a
        single string suitable for model input.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list with 'role' and 'content' keys.

        Returns
        -------
        str
            Formatted prompt string with all messages concatenated.
        """
        prompt = self.generate_prompt_turns(messages)
        return "".join(turn["content"] for turn in prompt)
