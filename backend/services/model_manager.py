from enum import Enum
from typing import Dict, Optional, List
import asyncio
from pydantic import BaseModel
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime

class ModelProvider(str, Enum):
    CLAUDE = "claude"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class ModelConfig(BaseModel):
    provider: ModelProvider
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    is_active: bool = True
    last_used: datetime = datetime.now()
    error_count: int = 0
    avg_latency: float = 0.0

class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.current_model: Optional[str] = None
        self.client = httpx.AsyncClient()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default models
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize default model configurations"""
        self.models = {
            "claude-3-sonnet": ModelConfig(
                provider=ModelProvider.CLAUDE,
                model_id="claude-3-sonnet",
                api_key="your_claude_api_key"  # Load from env
            ),
            "mistral-medium": ModelConfig(
                provider=ModelProvider.MISTRAL,
                model_id="mistral-medium",
                api_key="your_mistral_api_key"  # Load from env
            ),
            "huggingface-mixtral": ModelConfig(
                provider=ModelProvider.HUGGINGFACE,
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                api_key="your_hf_api_key"  # Load from env
            ),
            "ollama-local": ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_id="llama2",
                api_base="http://localhost:11434"
            )
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_claude(self, model_config: ModelConfig, prompt: str) -> str:
        """Query Claude API with retry mechanism"""
        try:
            client = anthropic.AsyncAnthropic(api_key=model_config.api_key)
            response = await client.messages.create(
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            model_config.error_count += 1
            raise

    async def query_mistral(self, model_config: ModelConfig, prompt: str) -> str:
        """Query Mistral API"""
        try:
            headers = {"Authorization": f"Bearer {model_config.api_key}"}
            response = await self.client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_config.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature
                }
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Mistral API error: {str(e)}")
            model_config.error_count += 1
            raise

    async def query_huggingface(self, model_config: ModelConfig, prompt: str) -> str:
        """Query Hugging Face API"""
        try:
            headers = {"Authorization": f"Bearer {model_config.api_key}"}
            response = await self.client.post(
                f"https://api-inference.huggingface.co/models/{model_config.model_id}",
                headers=headers,
                json={"inputs": prompt}
            )
            return response.json()[0]["generated_text"]
        except Exception as e:
            self.logger.error(f"Hugging Face API error: {str(e)}")
            model_config.error_count += 1
            raise

    async def query_ollama(self, model_config: ModelConfig, prompt: str) -> str:
        """Query local Ollama instance"""
        try:
            response = await self.client.post(
                f"{model_config.api_base}/api/generate",
                json={
                    "model": model_config.model_id,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()["response"]
        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            model_config.error_count += 1
            raise

    async def get_response(self, prompt: str) -> str:
        """Get response from the best available model"""
        if not self.current_model:
            self.current_model = self._select_best_model()

        model_config = self.models[self.current_model]
        start_time = datetime.now()

        try:
            response = await self._query_model(model_config, prompt)
            
            # Update metrics
            model_config.last_used = datetime.now()
            model_config.avg_latency = (datetime.now() - start_time).total_seconds()
            return response

        except Exception as e:
            self.logger.error(f"Error with model {self.current_model}: {str(e)}")
            # Switch to next best model
            self.current_model = self._select_best_model(exclude=[self.current_model])
            if self.current_model:
                return await self.get_response(prompt)
            raise Exception("All models failed")

    async def _query_model(self, model_config: ModelConfig, prompt: str) -> str:
        """Route query to appropriate model API"""
        if model_config.provider == ModelProvider.CLAUDE:
            return await self.query_claude(model_config, prompt)
        elif model_config.provider == ModelProvider.MISTRAL:
            return await self.query_mistral(model_config, prompt)
        elif model_config.provider == ModelProvider.HUGGINGFACE:
            return await self.query_huggingface(model_config, prompt)
        elif model_config.provider == ModelProvider.OLLAMA:
            return await self.query_ollama(model_config, prompt)
        raise ValueError(f"Unknown provider: {model_config.provider}")

    def _select_best_model(self, exclude: List[str] = None) -> Optional[str]:
        """Select the best available model based on metrics"""
        if exclude is None:
            exclude = []

        available_models = {
            name: config for name, config in self.models.items()
            if config.is_active and name not in exclude
        }

        if not available_models:
            return None

        # Simple selection strategy based on error count and latency
        return min(
            available_models.items(),
            key=lambda x: (x[1].error_count, x[1].avg_latency)
        )[0]

    def add_model(self, name: str, config: ModelConfig):
        """Add a new model configuration"""
        self.models[name] = config

    def remove_model(self, name: str):
        """Remove a model configuration"""
        if name in self.models:
            del self.models[name]
            if self.current_model == name:
                self.current_model = self._select_best_model()

    def update_model_status(self, name: str, is_active: bool):
        """Update model availability status"""
        if name in self.models:
            self.models[name].is_active = is_active
            if not is_active and self.current_model == name:
                self.current_model = self._select_best_model(exclude=[name]) 