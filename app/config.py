from dataclasses import dataclass


@dataclass
class Settings:
    default_mode: str = "sample"  # or "full"


settings = Settings()
