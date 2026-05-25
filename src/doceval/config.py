"""Application settings (loaded from environment / .env)."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. All values can be overridden by env vars
    prefixed with ``INTSIG_EVAL_`` (see ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="INTSIG_EVAL_",
        extra="ignore",
    )

    # --- Azure OpenAI (vision verifier) -----------------------------------
    # Environment-specific values live in `.env` (see `.env.example`).
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2025-04-01-preview"

    # Tenant that owns the Azure OpenAI resource above. Pinning this avoids
    # ``DefaultAzureCredential`` picking up an unrelated VS Code / managed-
    # identity login from another tenant and producing
    # ``Tenant ... does not match resource tenant`` 400s.
    azure_tenant_id: str = ""

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

    @property
    def image_dir(self) -> Path:
        return self.data_root / "image" / "source"

    @property
    def md_root(self) -> Path:
        return self.data_root / "MD"

    @property
    def ocr_cache_dir(self) -> Path:
        return self.data_root / ".cache" / "ocr"

    @property
    def out_root(self) -> Path:
        return self.data_root / self.output_dir


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
