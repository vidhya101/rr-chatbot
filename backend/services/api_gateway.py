import os
import json
import logging
import httpx
from typing import Dict, List, Union, Optional, Any, Callable
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime, timedelta
import jwt
import uuid
import asyncio
from functools import wraps
import redis
from prometheus_client import Counter, Histogram, generate_latest
from circuitbreaker import circuit
import aiohttp
from typing import Dict, List, Set
import time
from dataclasses import dataclass
from enum import Enum

# ... existing code ...

# New configuration models
class ServiceConfig(BaseModel):
    name: str
    url: str
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    rate_limit: Optional[int] = None
    required_roles: Set[str] = set()

class RateLimitConfig(BaseModel):
    requests_per_minute: int
    burst_size: int

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    recovery_timeout: int
    half_open_timeout: int

# Enhanced service configuration
ENHANCED_SERVICE_CONFIG = {
    "auth": ServiceConfig(
        name="auth",
        url="http://localhost:8000",
        timeout=10,
        retry_count=2,
        required_roles={"admin"}
    ),
    "data_processing": ServiceConfig(
        name="data_processing",
        url="http://localhost:8001",
        timeout=30,
        rate_limit=100
    ),
    "ml": ServiceConfig(
        name="ml",
        url="http://localhost:8002",
        timeout=60,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=120
    ),
    "visualization": ServiceConfig(
        name="visualization",
        url="http://localhost:8003",
        timeout=20
    ),
    "llm": ServiceConfig(
        name="llm",
        url="http://localhost:8004",
        timeout=45,
        rate_limit=50
    )
}

# Initialize Redis for rate limiting and caching
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'gateway_requests_total',
    'Total requests processed by the gateway',
    ['service', 'endpoint', 'method', 'status']
)

RESPONSE_TIME = Histogram(
    'gateway_response_time_seconds',
    'Response time in seconds',
    ['service', 'endpoint']
)

# Circuit breaker state
circuit_breaker_states = {
    service: {
        "state": CircuitBreakerState.CLOSED,
        "failures": 0,
        "last_failure_time": 0,
        "last_success_time": 0
    }
    for service in ENHANCED_SERVICE_CONFIG
}

# Utility functions for advanced features
async def check_rate_limit(service: str, user_id: str) -> bool:
    """Check if request is within rate limits"""
    if service not in ENHANCED_SERVICE_CONFIG:
        return True
    
    config = ENHANCED_SERVICE_CONFIG[service]
    if not config.rate_limit:
        return True
    
    key = f"rate_limit:{service}:{user_id}"
    current = redis_client.get(key)
    
    if not current:
        redis_client.setex(key, 60, 1)
        return True
    
    if int(current) >= config.rate_limit:
        return False
    
    redis_client.incr(key)
    return True

async def check_circuit_breaker(service: str) -> bool:
    """Check if circuit breaker allows the request"""
    if service not in circuit_breaker_states:
        return True
    
    state = circuit_breaker_states[service]
    config = ENHANCED_SERVICE_CONFIG[service]
    current_time = time.time()
    
    if state["state"] == CircuitBreakerState.OPEN:
        if current_time - state["last_failure_time"] > config.circuit_breaker_timeout:
            state["state"] = CircuitBreakerState.HALF_OPEN
        else:
            return False
    
    return True

async def update_circuit_breaker(service: str, success: bool):
    """Update circuit breaker state based on request result"""
    if service not in circuit_breaker_states:
        return
    
    state = circuit_breaker_states[service]
    config = ENHANCED_SERVICE_CONFIG[service]
    current_time = time.time()
    
    if success:
        if state["state"] == CircuitBreakerState.HALF_OPEN:
            state["state"] = CircuitBreakerState.CLOSED
        state["failures"] = 0
        state["last_success_time"] = current_time
    else:
        state["failures"] += 1
        state["last_failure_time"] = current_time
        
        if state["failures"] >= config.circuit_breaker_threshold:
            state["state"] = CircuitBreakerState.OPEN

class ServiceMetrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.last_error_time = None
        self.last_success_time = None

service_metrics = {service: ServiceMetrics() for service in ENHANCED_SERVICE_CONFIG}

# Cache configuration
CACHE_TTL = {
    "auth": 300,  # 5 minutes
    "data_processing": 600,  # 10 minutes
    "ml": 1800,  # 30 minutes
    "visualization": 900,  # 15 minutes
    "llm": 60  # 1 minute
}

async def get_cached_response(service: str, path: str, method: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Get cached response if available"""
    if method.lower() != "get":
        return None
    
    cache_key = f"cache:{service}:{path}:{json.dumps(params or {})}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    return None

async def cache_response(service: str, path: str, method: str, params: Optional[Dict], response: Dict):
    """Cache the response if appropriate"""
    if method.lower() != "get":
        return
    
    if service not in CACHE_TTL:
        return
    
    cache_key = f"cache:{service}:{path}:{json.dumps(params or {})}"
    redis_client.setex(cache_key, CACHE_TTL[service], json.dumps(response))

async def forward_request(
    service: str,
    path: str,
    method: str,
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    current_user: Optional[User] = None,
    params: Optional[Dict] = None,
    stream: bool = False
) -> Any:
    """
    Enhanced request forwarding with circuit breaker, rate limiting, caching,
    retries, and metrics tracking
    """
    if service not in ENHANCED_SERVICE_CONFIG:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
    
    config = ENHANCED_SERVICE_CONFIG[service]
    start_time = time.time()
    
    try:
        # Check rate limits
        if current_user and not await check_rate_limit(service, current_user.user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Check circuit breaker
        if not await check_circuit_breaker(service):
            raise HTTPException(status_code=503, detail=f"Service '{service}' is temporarily unavailable")
        
        # Check role-based access
        if config.required_roles and not (current_user and current_user.role in config.required_roles):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Check cache for GET requests
        if method.lower() == "get" and not stream:
            cached_response = await get_cached_response(service, path, method, params)
            if cached_response:
                # Update metrics
                REQUEST_COUNT.labels(service=service, endpoint=path, method=method, status="hit").inc()
                RESPONSE_TIME.labels(service=service, endpoint=path).observe(time.time() - start_time)
                return cached_response
        
        url = f"{config.url}{path}"
        
        # Prepare headers
        request_headers = headers or {}
        if current_user:
            request_headers.update({
                "X-User-ID": current_user.user_id,
                "X-User-Role": current_user.role,
                "X-Request-ID": str(uuid.uuid4()),
                "X-Correlation-ID": request_headers.get("X-Correlation-ID", str(uuid.uuid4()))
            })
        
        # Initialize retry counter
        retry_count = 0
        last_exception = None
        
        while retry_count <= config.retry_count:
            try:
                async with httpx.AsyncClient() as client:
                    if stream:
                        response = await client.stream(
                            method,
                            url,
                            json=data,
                            headers=request_headers,
                            params=params,
                            timeout=config.timeout
                        )
                        
                        # Update metrics and circuit breaker
                        REQUEST_COUNT.labels(service=service, endpoint=path, method=method, status=response.status_code).inc()
                        RESPONSE_TIME.labels(service=service, endpoint=path).observe(time.time() - start_time)
                        await update_circuit_breaker(service, response.status_code < 500)
                        
                        return response
                    else:
                        if method.lower() == "get":
                            response = await client.get(
                                url,
                                headers=request_headers,
                                params=params,
                                timeout=config.timeout
                            )
                        elif method.lower() == "post":
                            response = await client.post(
                                url,
                                json=data,
                                headers=request_headers,
                                params=params,
                                timeout=config.timeout
                            )
                        elif method.lower() == "put":
                            response = await client.put(
                                url,
                                json=data,
                                headers=request_headers,
                                params=params,
                                timeout=config.timeout
                            )
                        elif method.lower() == "delete":
                            response = await client.delete(
                                url,
                                headers=request_headers,
                                params=params,
                                timeout=config.timeout
                            )
                        else:
                            raise HTTPException(status_code=405, detail=f"Method {method} not allowed")
                        
                        # Update metrics
                        REQUEST_COUNT.labels(service=service, endpoint=path, method=method, status=response.status_code).inc()
                        RESPONSE_TIME.labels(service=service, endpoint=path).observe(time.time() - start_time)
                        
                        # Check response status
                        if response.status_code >= 400:
                            error_detail = response.text
                            try:
                                error_json = response.json()
                                if "detail" in error_json:
                                    error_detail = error_json["detail"]
                            except:
                                pass
                            
                            # Update circuit breaker for 5xx errors
                            if response.status_code >= 500:
                                await update_circuit_breaker(service, False)
                                if retry_count < config.retry_count:
                                    retry_count += 1
                                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                                    continue
                            
                            raise HTTPException(status_code=response.status_code, detail=error_detail)
                        
                        # Update circuit breaker for success
                        await update_circuit_breaker(service, True)
                        
                        # Parse and cache response
                        response_data = response.json()
                        if method.lower() == "get":
                            await cache_response(service, path, method, params, response_data)
                        
                        return response_data
            
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                last_exception = e
                if retry_count < config.retry_count:
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    # Update circuit breaker and metrics for final failure
                    await update_circuit_breaker(service, False)
                    REQUEST_COUNT.labels(service=service, endpoint=path, method=method, status="error").inc()
                    RESPONSE_TIME.labels(service=service, endpoint=path).observe(time.time() - start_time)
                    logger.error(f"Error forwarding request to {service}: {str(e)}")
                    raise HTTPException(status_code=503, detail=f"Service '{service}' is unavailable")
        
        # If we've exhausted retries
        if last_exception:
            raise HTTPException(status_code=503, detail=f"Service '{service}' is unavailable after {config.retry_count} retries")
    
    except Exception as e:
        # Update metrics for unexpected errors
        REQUEST_COUNT.labels(service=service, endpoint=path, method=method, status="error").inc()
        RESPONSE_TIME.labels(service=service, endpoint=path).observe(time.time() - start_time)
        logger.error(f"Unexpected error in forward_request: {str(e)}")
        raise

@app.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """
    Get Prometheus metrics
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return Response(generate_latest(), media_type="text/plain")

@app.get("/service-status")
async def get_service_status(current_user: User = Depends(get_current_user)):
    """
    Get detailed status of all services including circuit breaker state,
    rate limits, and recent metrics
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    status = {}
    current_time = time.time()
    
    for service, config in ENHANCED_SERVICE_CONFIG.items():
        circuit_state = circuit_breaker_states[service]
        metrics = service_metrics[service]
        
        # Get current rate limit counts
        rate_limit_key = f"rate_limit:{service}:{current_user.user_id}"
        current_rate = redis_client.get(rate_limit_key)
        
        status[service] = {
            "config": {
                "url": config.url,
                "timeout": config.timeout,
                "retry_count": config.retry_count,
                "rate_limit": config.rate_limit,
                "circuit_breaker_threshold": config.circuit_breaker_threshold
            },
            "circuit_breaker": {
                "state": circuit_state["state"].value,
                "failures": circuit_state["failures"],
                "last_failure": datetime.fromtimestamp(circuit_state["last_failure_time"]).isoformat() if circuit_state["last_failure_time"] else None,
                "last_success": datetime.fromtimestamp(circuit_state["last_success_time"]).isoformat() if circuit_state["last_success_time"] else None
            },
            "rate_limiting": {
                "current_usage": int(current_rate) if current_rate else 0,
                "limit": config.rate_limit
            },
            "metrics": {
                "request_count": metrics.request_count,
                "error_count": metrics.error_count,
                "avg_response_time": metrics.total_response_time / metrics.request_count if metrics.request_count > 0 else 0,
                "last_error": metrics.last_error_time.isoformat() if metrics.last_error_time else None,
                "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None
            }
        }
    
    return status

@app.post("/service-config/{service}")
async def update_service_config(
    service: str,
    config: ServiceConfig,
    current_user: User = Depends(get_current_user)
):
    """
    Update configuration for a specific service
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if service not in ENHANCED_SERVICE_CONFIG:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
    
    # Update configuration
    ENHANCED_SERVICE_CONFIG[service] = config
    
    # Reset circuit breaker state
    circuit_breaker_states[service] = {
        "state": CircuitBreakerState.CLOSED,
        "failures": 0,
        "last_failure_time": 0,
        "last_success_time": 0
    }
    
    return {"message": f"Configuration updated for service '{service}'"}

@app.post("/circuit-breaker/{service}/reset")
async def reset_circuit_breaker(
    service: str,
    current_user: User = Depends(get_current_user)
):
    """
    Reset circuit breaker state for a specific service
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if service not in circuit_breaker_states:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
    
    circuit_breaker_states[service] = {
        "state": CircuitBreakerState.CLOSED,
        "failures": 0,
        "last_failure_time": 0,
        "last_success_time": 0
    }
    
    return {"message": f"Circuit breaker reset for service '{service}'"}

@app.post("/cache/clear/{service}")
async def clear_service_cache(
    service: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear cache for a specific service
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if service not in ENHANCED_SERVICE_CONFIG:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
    
    # Clear all cache keys for the service
    pattern = f"cache:{service}:*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
    
    return {"message": f"Cache cleared for service '{service}'"}

@app.get("/request-logs")
async def get_request_logs(
    service: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    status_code: Optional[int] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """
    Get recent request logs with filtering options
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # This is a simplified implementation. In a real system,
    # you would store logs in a proper database or log aggregation service.
    logs = []
    
    # Read from log file
    with open("api_gateway.log", "r") as f:
        for line in f:
            try:
                # Parse log entry
                log_entry = json.loads(line)
                
                # Apply filters
                if service and log_entry.get("service") != service:
                    continue
                if start_time and datetime.fromisoformat(log_entry["timestamp"]) < start_time:
                    continue
                if end_time and datetime.fromisoformat(log_entry["timestamp"]) > end_time:
                    continue
                if status_code and log_entry.get("status_code") != status_code:
                    continue
                
                logs.append(log_entry)
                
                if len(logs) >= limit:
                    break
            except:
                continue
    
    return logs

# ... rest of the existing code ... 