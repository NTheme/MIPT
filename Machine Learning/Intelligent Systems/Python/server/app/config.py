# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 May 2025, 1:25 AM
#   * File    : config.py
#   * Project : Python
# --========================================-- #


from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_dir: Path
    num_cores: int
    max_loaded_models: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
