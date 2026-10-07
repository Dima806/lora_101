"""Typed settings loaded from ``config/settings.yaml`` via pydantic-settings.

Nothing in ``src/`` or the notebooks should hardcode ranks, alpha, target layers,
step counts, or seeds; read them from :func:`get_settings` instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class LoRASettings(BaseModel):
    """Adapter hyperparameters."""

    r: int = 8
    alpha: float = 16.0
    target_modules: list[str] = Field(default_factory=lambda: ["linear"])


class ToySettings(BaseModel):
    """The transparent from-scratch task with a planted low-rank update."""

    dim: int = 128
    out_dim: int = 128
    true_rank: int = 4
    n_samples: int = 256
    noise: float = 0.0


class TrainSettings(BaseModel):
    """Full-batch CPU training loop."""

    steps: int = 800
    lr: float = 0.03


class HeadlineSettings(BaseModel):
    """Dimensions for the counted (not trained) <1% parameter claim."""

    dim: int = 1024
    r: int = 4


class SmallLMSettings(BaseModel):
    """The tiny HF model used only for the style-vs-facts boundary."""

    model_name: str = "sshleifer/tiny-gpt2"
    max_steps: int = 300
    lr: float = 5e-4


class Settings(BaseSettings):
    """Top-level project configuration."""

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        env_prefix="LORA101_",
        extra="ignore",
    )

    seed: int = 0
    ranks: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    lora: LoRASettings = Field(default_factory=LoRASettings)
    toy: ToySettings = Field(default_factory=ToySettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    headline: HeadlineSettings = Field(default_factory=HeadlineSettings)
    small_lm: SmallLMSettings = Field(default_factory=SmallLMSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # init kwargs win (handy for tests), then the YAML file, then env vars.
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
        )


def get_settings() -> Settings:
    """Load settings from ``config/settings.yaml`` (with env/init overrides)."""
    return Settings()
