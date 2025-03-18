import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def generate_claude_response(messages: List[Dict[str, Any]], model: str = "claude-3-opus-20240229") -> str:
    """Generate a response using Claude API"""
    if not CLAUDE_API_KEY:
        logger.error("Claude API key not found")
        return "Claude API key not configured. Please set the CLAUDE_API_KEY environment variable."
    
    try:
        # Convert messages to Claude format
        claude_messages = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System messages are handled differently in Claude
                system_content = content
                continue
            
            claude_role = "user" if role == "user" else "assistant"
            claude_messages.append({"role": claude_role, "content": content})
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": claude_messages,
            "max_tokens": 4000,
            "temperature": 0.7,
            "system": system_content if 'system_content' in locals() else ""
        }
        
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        response = requests.post(
            CLAUDE_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        # Check for successful response
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("content", [{"text": "No response from Claude"}])[0].get("text", "")
        else:
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return f"Error from Claude API: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Error generating Claude response: {str(e)}")
        return f"Error generating response: {str(e)}"

def is_claude_available() -> bool:
    """Check if Claude API is available"""
    return bool(CLAUDE_API_KEY) 