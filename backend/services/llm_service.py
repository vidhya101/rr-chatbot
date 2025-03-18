import os
import json
import logging
import requests
import httpx
from typing import Dict, List, Union, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import asyncio
import uuid
import aiohttp
import re
from enum import Enum
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ... existing code ...

# New configuration models
class PromptTemplate(BaseModel):
    name: str
    template: str
    variables: List[str]
    description: Optional[str]

class ModelConfig(BaseModel):
    name: str
    provider: str
    context_window: int
    max_tokens: int
    capabilities: List[str]
    cost_per_token: float
    recommended_temperature: float

class FunctionCall(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ChatFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    required: List[str]

class AdvancedChatRequest(ChatRequest):
    system_prompt: Optional[str]
    functions: Optional[List[ChatFunction]]
    prompt_template: Optional[str]
    stream_tokens: Optional[bool] = False
    include_sources: Optional[bool] = False
    semantic_cache: Optional[bool] = False
    timeout: Optional[int] = 30

class ModelMetrics(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float
    response_time: float

# Utility functions for advanced features
def create_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Create embeddings for text using the specified model"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling to get text embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

async def semantic_search(query: str, conversation_history: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Search conversation history semantically"""
    query_embedding = create_embedding(query)
    
    # Create embeddings for all messages in history
    history_embeddings = []
    for msg in conversation_history:
        if isinstance(msg["content"], str):
            embedding = create_embedding(msg["content"])
            history_embeddings.append(embedding)
    
    if not history_embeddings:
        return []
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, np.vstack(history_embeddings))[0]
    
    # Get top-k most similar messages
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [conversation_history[i] for i in top_indices]

class ResponseCache:
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.embeddings = {}
    
    async def get(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """Get cached response if similar query exists"""
        if not self.cache:
            return None
        
        query_embedding = create_embedding(query)
        
        # Find most similar cached query
        max_similarity = 0
        best_response = None
        
        for cached_query, embedding in self.embeddings.items():
            similarity = cosine_similarity(query_embedding, embedding)[0][0]
            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                best_response = self.cache[cached_query]
        
        return best_response
    
    async def add(self, query: str, response: str):
        """Add query-response pair to cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_query = next(iter(self.cache))
            del self.cache[oldest_query]
            del self.embeddings[oldest_query]
        
        self.cache[query] = response
        self.embeddings[query] = create_embedding(query)

# Initialize response cache
response_cache = ResponseCache()

# ... rest of the existing code ... 

@app.post("/advanced_chat", response_model=ChatResponse)
async def advanced_chat(request: AdvancedChatRequest):
    """
    Advanced chat endpoint with additional features like function calling,
    semantic caching, and prompt templates
    """
    try:
        start_time = datetime.now()
        
        # Check semantic cache if enabled
        if request.semantic_cache:
            cached_response = await response_cache.get(request.messages[-1].content)
            if cached_response:
                return ChatResponse(
                    conversation_id=request.conversation_id or f"conversation_{uuid.uuid4().hex[:8]}",
                    message=cached_response,
                    model=request.model,
                    provider=request.provider,
                    created_at=datetime.now().isoformat()
                )
        
        # Apply prompt template if specified
        if request.prompt_template:
            template = next((t for t in prompt_templates if t.name == request.prompt_template), None)
            if template:
                # Extract variables from the last user message
                variables = {}
                for var in template.variables:
                    match = re.search(f"{{{var}}}", request.messages[-1].content)
                    if match:
                        variables[var] = match.group(1)
                
                # Apply template
                formatted_prompt = template.template.format(**variables)
                request.messages[-1].content = formatted_prompt
        
        # Add system prompt if specified
        if request.system_prompt:
            request.messages.insert(0, Message(role="system", content=request.system_prompt))
        
        # Generate response with function calling if specified
        if request.functions:
            response_message = await generate_llm_response_with_functions(
                provider=request.provider,
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                functions=request.functions
            )
        else:
            response_message = await generate_llm_response(
                provider=request.provider,
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        # Calculate metrics
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Add to semantic cache if enabled
        if request.semantic_cache:
            await response_cache.add(request.messages[-1].content, response_message)
        
        return ChatResponse(
            conversation_id=request.conversation_id or f"conversation_{uuid.uuid4().hex[:8]}",
            message=response_message,
            model=request.model,
            provider=request.provider,
            created_at=end_time.isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in advanced chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/batch_process")
async def batch_process(requests: List[ChatRequest], background_tasks: BackgroundTasks):
    """
    Process multiple chat requests in parallel
    """
    try:
        async def process_request(req: ChatRequest):
            try:
                return await chat(req)
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                return {"error": str(e)}
        
        # Process requests in parallel
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": results,
            "successful": len([r for r in results if not isinstance(r, Exception)]),
            "failed": len([r for r in results if isinstance(r, Exception)])
        }
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")

@app.post("/analyze_conversation/{conversation_id}")
async def analyze_conversation(conversation_id: str):
    """
    Analyze a conversation for insights
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        conversation = conversations[conversation_id]
        messages = conversation["messages"]
        
        # Calculate basic metrics
        message_count = len(messages)
        user_messages = len([m for m in messages if m["role"] == "user"])
        assistant_messages = len([m for m in messages if m["role"] == "assistant"])
        
        # Calculate average message length
        avg_user_length = np.mean([len(m["content"]) for m in messages if m["role"] == "user"])
        avg_assistant_length = np.mean([len(m["content"]) for m in messages if m["role"] == "assistant"])
        
        # Identify common topics using embeddings
        topics = []
        if message_count > 0:
            # Create embeddings for all messages
            embeddings = [create_embedding(m["content"]) for m in messages]
            
            # Cluster embeddings to identify topics
            from sklearn.cluster import KMeans
            n_clusters = min(5, message_count)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(np.vstack(embeddings))
            
            # Get representative messages for each cluster
            for i in range(n_clusters):
                cluster_messages = [m["content"] for j, m in enumerate(messages) if clusters[j] == i]
                if cluster_messages:
                    topics.append(cluster_messages[0][:100] + "...")
        
        return {
            "metrics": {
                "total_messages": message_count,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "avg_user_message_length": avg_user_length,
                "avg_assistant_message_length": avg_assistant_length
            },
            "topics": topics,
            "duration": (datetime.fromisoformat(conversation["updated_at"]) - 
                        datetime.fromisoformat(conversation["created_at"])).total_seconds(),
            "active_hours": len(set(datetime.fromisoformat(m["timestamp"]).hour 
                                  for m in messages))
        }
    
    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing conversation: {str(e)}")

@app.post("/compare_models")
async def compare_models(request: ChatRequest, models: List[str]):
    """
    Compare responses from multiple models for the same input
    """
    try:
        async def get_model_response(model: str):
            try:
                start_time = datetime.now()
                response = await generate_llm_response(
                    provider=request.provider,
                    model=model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                end_time = datetime.now()
                
                return {
                    "model": model,
                    "response": response,
                    "response_time": (end_time - start_time).total_seconds()
                }
            except Exception as e:
                return {
                    "model": model,
                    "error": str(e)
                }
        
        # Get responses from all models in parallel
        tasks = [get_model_response(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        # Calculate response similarity matrix
        similarity_matrix = {}
        successful_responses = [r for r in results if "response" in r]
        
        if len(successful_responses) > 1:
            embeddings = [create_embedding(r["response"]) for r in successful_responses]
            similarities = cosine_similarity(np.vstack(embeddings))
            
            for i, r1 in enumerate(successful_responses):
                similarity_matrix[r1["model"]] = {
                    r2["model"]: float(similarities[i][j])
                    for j, r2 in enumerate(successful_responses)
                }
        
        return {
            "results": results,
            "similarity_matrix": similarity_matrix,
            "successful": len(successful_responses),
            "failed": len(results) - len(successful_responses)
        }
    
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing models: {str(e)}")

async def generate_llm_response_with_functions(
    provider: str,
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    functions: List[ChatFunction]
) -> str:
    """
    Generate a response from the LLM with function calling capabilities
    """
    try:
        if provider == "claude":
            return await generate_claude_response_with_functions(
                model, messages, temperature, max_tokens, functions
            )
        elif provider == "mistral":
            return await generate_mistral_response_with_functions(
                model, messages, temperature, max_tokens, functions
            )
        else:
            raise ValueError(f"Function calling not supported for provider: {provider}")
    
    except Exception as e:
        logger.error(f"Error in function calling: {str(e)}")
        raise

async def generate_claude_response_with_functions(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    functions: List[ChatFunction]
) -> str:
    """
    Generate a response from Claude AI with function calling
    """
    try:
        api_key = CONFIG["claude"]["api_key"]
        
        # Convert messages and functions to Claude's format
        claude_messages = []
        for msg in messages:
            role = "user" if msg.role in ["user", "human"] else "assistant"
            claude_messages.append({"role": role, "content": msg.content})
        
        # Add function descriptions to system message
        function_descriptions = "\n\n".join([
            f"Function: {func.name}\nDescription: {func.description}\n"
            f"Parameters: {json.dumps(func.parameters, indent=2)}"
            for func in functions
        ])
        
        system_message = (
            "You have access to the following functions:\n\n"
            f"{function_descriptions}\n\n"
            "To use a function, respond with a JSON object containing 'function_call' "
            "with the function name and parameters."
        )
        
        claude_messages.insert(0, {"role": "system", "content": system_message})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": model,
                    "messages": claude_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Claude API error: {response.status} - {error_text}")
                
                result = await response.json()
                response_text = result["content"][0]["text"]
                
                # Check if response contains a function call
                try:
                    response_json = json.loads(response_text)
                    if "function_call" in response_json:
                        function_call = response_json["function_call"]
                        # Here you would implement the actual function execution
                        return f"Function call requested: {json.dumps(function_call)}"
                except json.JSONDecodeError:
                    pass
                
                return response_text
    
    except Exception as e:
        logger.error(f"Error in Claude function calling: {str(e)}")
        raise

async def generate_mistral_response_with_functions(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    functions: List[ChatFunction]
) -> str:
    """
    Generate a response from Mistral AI with function calling
    """
    try:
        api_key = CONFIG["mistral"]["api_key"]
        
        # Convert messages and functions to Mistral's format
        mistral_messages = []
        for msg in messages:
            role = "user" if msg.role in ["user", "human"] else "assistant"
            mistral_messages.append({"role": role, "content": msg.content})
        
        # Add function descriptions to system message
        function_descriptions = "\n\n".join([
            f"Function: {func.name}\nDescription: {func.description}\n"
            f"Parameters: {json.dumps(func.parameters, indent=2)}"
            for func in functions
        ])
        
        system_message = (
            "You have access to the following functions:\n\n"
            f"{function_descriptions}\n\n"
            "To use a function, respond with a JSON object containing 'function_call' "
            "with the function name and parameters."
        )
        
        mistral_messages.insert(0, {"role": "system", "content": system_message})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": mistral_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Mistral API error: {response.status} - {error_text}")
                
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Check if response contains a function call
                try:
                    response_json = json.loads(response_text)
                    if "function_call" in response_json:
                        function_call = response_json["function_call"]
                        # Here you would implement the actual function execution
                        return f"Function call requested: {json.dumps(function_call)}"
                except json.JSONDecodeError:
                    pass
                
                return response_text
    
    except Exception as e:
        logger.error(f"Error in Mistral function calling: {str(e)}")
        raise

# Additional utility functions
def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in a text string for a specific model
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}")
        # Fallback to approximate token count
        return len(text.split())

def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an API call based on token usage
    """
    try:
        if provider not in CONFIG or "cost_per_token" not in CONFIG[provider]:
            return 0.0
        
        cost_per_token = CONFIG[provider]["cost_per_token"]
        total_tokens = prompt_tokens + completion_tokens
        
        return total_tokens * cost_per_token
    except Exception as e:
        logger.warning(f"Error estimating cost: {str(e)}")
        return 0.0

def validate_response(response: str, expected_format: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Validate an LLM response against expected format
    """
    try:
        if not response:
            return False, "Empty response"
        
        if expected_format:
            try:
                response_json = json.loads(response)
                for key, value_type in expected_format.items():
                    if key not in response_json:
                        return False, f"Missing required key: {key}"
                    if not isinstance(response_json[key], value_type):
                        return False, f"Invalid type for key {key}"
                return True, "Valid response"
            except json.JSONDecodeError:
                return False, "Invalid JSON format"
        
        return True, "Valid response"
    except Exception as e:
        logger.error(f"Error validating response: {str(e)}")
        return False, str(e)

# Initialize prompt templates
prompt_templates = [
    PromptTemplate(
        name="summarize",
        template="Please provide a concise summary of the following text:\n\n{text}",
        variables=["text"],
        description="Generate a summary of a given text"
    ),
    PromptTemplate(
        name="analyze",
        template="Please analyze the following {topic} and provide key insights:\n\n{content}",
        variables=["topic", "content"],
        description="Analyze content and provide insights"
    ),
    PromptTemplate(
        name="compare",
        template="Please compare and contrast the following items:\n\nItem 1: {item1}\nItem 2: {item2}",
        variables=["item1", "item2"],
        description="Compare two items and highlight differences"
    )
]

# ... rest of the existing code ... 