"""Application settings (loaded from environment / .env)."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ModelSource = Literal["aoai", "copilot"]


class Settings(BaseSettings):
    """Runtime configuration. All values can be overridden by env vars
    prefixed with ``DOCEVAL_`` (see ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCEVAL_",
        extra="ignore",
        # Pydantic v2 warns when fields start with ``model_``. Disable that
        # so ``model_source`` / ``model_name`` are legal field names.
        protected_namespaces=(),
        # Allow population by both field name and any aliases declared on
        # fields (e.g. the bare ``COPILOT_OAUTH_TOKEN`` env var).
        populate_by_name=True,
    )

    # --- Model provider selection ----------------------------------------
    # ``aoai``    → Azure OpenAI (AAD via AzureCliCredential, deployment alias)
    # ``copilot`` → GitHub Copilot subscription (OAuth + dynamic API token)
    model_source: ModelSource = "aoai"

    # Unified model identifier. For ``aoai`` this is the deployment name on
    # the Azure OpenAI resource (e.g. ``gpt-5.4``). For ``copilot`` this is
    # the model id returned by Copilot's ``/models`` endpoint (e.g.
    # ``gpt-5.4``, ``gpt-5.5``, ``claude-opus-4.7``). Used by the GPT
    # markdown generator and as the fallback for the verifier.
    model_name: str = "gpt-5.4"

    # Override the model used by the vision verifier. Leave empty to reuse
    # ``model_name``. Handy when you want a stronger reasoning model to
    # adjudicate hallucinations while still transcribing with a cheaper one
    # (e.g. generator=``gpt-5.4`` + verifier=``claude-opus-4.7``).
    verifier_model: str = ""

    # --- Azure OpenAI (vision verifier) -----------------------------------
    # Environment-specific values live in `.env` (see `.env.example`).
    # Used only when ``model_source == "aoai"``.
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2025-04-01-preview"

    # Tenant that owns the Azure OpenAI resource above. Pinning this avoids
    # ``DefaultAzureCredential`` picking up an unrelated VS Code / managed-
    # identity login from another tenant and producing
    # ``Tenant ... does not match resource tenant`` 400s.
    azure_tenant_id: str = ""

    # --- GitHub Copilot (vision verifier) ---------------------------------
    # OAuth token (``ghu_…``) for the GitHub account whose Copilot
    # subscription should be used. Read from the plain ``COPILOT_OAUTH_TOKEN``
    # env var (no ``DOCEVAL_`` prefix) so it can be shared between projects.
    # Used only when ``model_source == "copilot"``. If empty, a device-flow
    # login is triggered on first use and the resulting token is written
    # back to ``.env``.
    copilot_oauth_token: str = Field(default="", alias="COPILOT_OAUTH_TOKEN")

    # --- Azure Document Intelligence (OCR) --------------------------------
    di_endpoint: str = ""
    di_key: str = ""

    # --- Layout ------------------------------------------------------------
    data_root: Path = Field(default_factory=lambda: Path.cwd())
    output_dir: Path = Path("output")

    # --- Algorithm knobs ---------------------------------------------------
    cluster_edit_distance: int = 1
    """Tokens whose pairwise edit distance is ≤ this are merged into the same cluster."""

    min_token_length: int = 3
    """Normalized tokens shorter than this are discarded."""

    verify_singletons: bool = True
    """Send single-source clusters to the vision verifier agent."""

    vision_max_dim: int = 2048
    """Longest image side (pixels) sent to vision models. Larger images are
    downscaled with Lanczos before encoding. 2048 matches the high-detail
    tile size used internally by OpenAI/Copilot vision endpoints — going
    higher just bloats the request without improving model quality, and
    Copilot's ``/responses`` enforces a hard request-body size limit
    (HTTP 413). Drop this to 1280 / 1024 if you still hit 413 on very
    large pages."""

    vision_jpeg_quality: int = 85
    """JPEG quality for images sent to vision models (1–95)."""

    @property
    def image_dir(self) -> Path:
        return self.data_root / "image" / "source"

    @property
    def md_root(self) -> Path:
        return self.data_root / "MD"

    @property
    def di_cache_dir(self) -> Path:
        """Where Azure Document Intelligence layout results are cached.

        Each entry is ``<stem>.<sha16>.json``. Lives under :attr:`md_root`
        so all model outputs (raw DI JSON + per-LLM markdown) stay together.
        """
        return self.md_root / "di"

    @property
    def gpt_md_dir(self) -> Path:
        """Where the on-demand GPT/vision-LLM markdown transcripts go.

        Folder name follows :attr:`model_name` so different models' outputs
        don't collide (e.g. ``MD/gpt-5.4/``, ``MD/gpt-5.5/``,
        ``MD/claude-opus-4.7/``).
        """
        return self.md_root / self.model_name

    # Subdirectories under MD/ that are NOT markdown sources (i.e. they
    # store auxiliary data like the DI cache JSON). Auto-discovery in
    # ``list_md_sources`` filters these out so they never end up wired into
    # a ``MarkdownReader``.
    reserved_md_subdirs: ClassVar[frozenset[str]] = frozenset({"di"})

    def list_md_sources(self) -> list[str]:
        """Return MD source folder names under :attr:`md_root`.

        Real markdown subdirs only — :attr:`reserved_md_subdirs` are excluded.
        """
        if not self.md_root.exists():
            return []
        return sorted(
            p.name
            for p in self.md_root.iterdir()
            if p.is_dir() and p.name not in self.reserved_md_subdirs
        )

    @property
    def out_root(self) -> Path:
        return self.data_root / self.output_dir

    @property
    def effective_verifier_model(self) -> str:
        """Model id used by the vision verifier.

        Falls back to :attr:`model_name` when ``verifier_model`` is empty.
        """
        return self.verifier_model or self.model_name


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
