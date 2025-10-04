from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseModel):
    user_name: str = "User"
    agent_name: str = "Agent"


class ModelSettings(BaseModel):
    fast_model: str = "mistral"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    agent: AgentSettings = AgentSettings()
    model: ModelSettings = ModelSettings()


settings = Settings()
