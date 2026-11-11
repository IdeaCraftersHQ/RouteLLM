"""Video preprocessing settings and the error a failed one raises."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

__all__ = ["LOCAL_TAG", "VideoConfig", "VideoError"]

#: The tag that marks a preprocessor (or an endpoint) as running locally.
LOCAL_TAG = "local"

_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")


class VideoError(Exception):
    """The video could not be preprocessed; the request must not go on.

    Attributes
    ----------
    status_code : int
        The HTTP status a server should answer with.
    public : str
        A message safe to show a client: it echoes nothing the service
        said beyond a bounded error code.
    detail : str
        What went wrong, for the server log only.
    code : str
        A short snake_case code, safe to show a client.
    """

    def __init__(self, status_code: int, public: str, detail: str = "", code: str = "video_error"):
        super().__init__(detail or public)
        self.status_code = status_code
        self.public = public
        self.detail = detail or public
        self.code = code


class VideoConfig(BaseModel):
    """One video preprocessor's settings, global or merged with a tier's.

    Attributes
    ----------
    mode : str
        `preprocess` replaces each video with frames and a transcript;
        `passthrough` forwards it unchanged, only to a leaf whose
        `capabilities.modalities_in` lists `video`; `refuse` answers 400.
    api_base : str, optional
        Base URL of the service. Required for mode `preprocess`.
    api_key_env : str, optional
        Name of the environment variable holding the key, read at call
        time. Never the key itself.
    timeout_ms : int
        Bound on one preprocessing call, in milliseconds.
    max_video_bytes : int
        Largest single video accepted, in decoded bytes.
    allow_urls : bool
        Remote URLs are refused whatever this says; it only selects the
        refusal message (fetching is not implemented).
    allow_public_base : bool
        DANGEROUS. Lets a `local`-tagged preprocessor name a public host.
    tags : list[str]
        Labels; `local` declares the service runs on this machine or
        network, which redact and block classes require.
    max_frames, layout, max_side, transcribe, language, ocr,
    scene_detection, require_transcript
        Options sent to the service with every video.
    max_transcript_chars : int
        Bound on the transcript (and, separately, the on-screen text)
        written into the message; the cut is noted in the text.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["preprocess", "passthrough", "refuse"] = "preprocess"
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_ms: int = Field(default=600_000, gt=0)
    max_video_bytes: int = Field(default=200 * 1024 * 1024, gt=0)
    allow_urls: StrictBool = False
    allow_public_base: StrictBool = False
    tags: list[str] = Field(default_factory=list)
    max_frames: int = Field(default=12, ge=1, le=64)
    layout: Literal["frames", "contact_sheet"] = "frames"
    max_side: int = Field(default=768, ge=256, le=2048)
    transcribe: StrictBool = True
    language: str | None = None
    ocr: StrictBool = False
    scene_detection: StrictBool = True
    require_transcript: StrictBool = False
    max_transcript_chars: int = Field(default=20_000, gt=0)

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        if value is not None and not _LANGUAGE.match(value):
            raise ValueError(f"language {value!r} is not a language code such as 'en' or 'pt-BR'")
        return value

    @field_validator("api_base")
    @classmethod
    def _validate_base(cls, value: str | None) -> str | None:
        if value is not None and not re.match(r"^https?://[^\s/]+", value):
            raise ValueError(f"api_base {value!r} must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _needs_base(self) -> "VideoConfig":
        if self.mode == "preprocess" and not self.api_base:
            raise ValueError("mode `preprocess` needs `api_base`")
        return self

    @property
    def is_local(self) -> bool:
        """Whether this preprocessor declares itself local."""
        return LOCAL_TAG in self.tags

    @property
    def active(self) -> bool:
        """Whether video parts are sent to a service under this config."""
        return self.mode == "preprocess"
