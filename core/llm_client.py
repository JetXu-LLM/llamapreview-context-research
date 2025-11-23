import os
import time
import logging
import requests
from typing import List, Dict, Optional, Any

# Setup logger
logger = logging.getLogger("LlamaPReview")

class DeepSeekClient:
    """
    Unified client for interacting with DeepSeek API.
    Supports both Chat (V3) and Reasoner (R1) models with robust error handling.
    """
    
    API_BASE = "https://api.deepseek.com"
    DEFAULT_TIMEOUT = 120
    REASONER_TIMEOUT = 460  # Longer timeout for R1/Reasoner models

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set in environment variables.")

    def query_with_template(
        self, 
        template: str, 
        parameters: Dict[str, Any], 
        model: str = "deepseek-reasoner",
        response_type: str = "json_object",
        temperature: float = 0.3
    ) -> str:
        """
        High-level method for Solution 2 (Search RAG).
        Formats a template and returns the content string directly.
        """
        try:
            content = template.format(**parameters)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting message template: {e}")

        messages = [{"role": "user", "content": content}]
        
        # Call the unified request method
        response_dict = self._execute_request(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": response_type}
        )

        if not response_dict:
            return ""

        try:
            # Extract content safely
            return response_dict['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"Malformed response structure: {e}")
            return ""

    def chat(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.3,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """
        High-level method for Solution 3 (Agentic RAG).
        Returns the full API response dict so the caller can track tokens/reasoning.
        """
        return self._execute_request(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            tools=tools
        )

    def _execute_request(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Core request logic with exponential backoff, logging, and timeout handling.
        """
        url = f"{self.API_BASE}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        # Determine timeout based on model type
        current_timeout = self.REASONER_TIMEOUT if "reasoner" in model else self.DEFAULT_TIMEOUT

        # Debug Logging
        logger.debug(f"DeepSeek Call: model={model}, temp={temperature}, format={response_format}")
        # Log first user message preview
        for msg in messages:
            if msg.get('role') == 'user':
                preview = str(msg.get('content', ''))[:100].replace('\n', ' ')
                logger.debug(f"  Input Preview: {preview}...")
                break

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=current_timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    self._log_success(result)
                    return result
                
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code >= 500:
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error ({response.status_code}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    # Client error (400, 401, etc) - Do not retry
                    error_detail = response.text[:500]
                    logger.error(f"API Client Error [{response.status_code}]: {error_detail}")
                    return None
            
            except requests.Timeout:
                logger.warning(f"Request timed out ({current_timeout}s) - Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries exceeded due to timeout.")
            
            except requests.RequestException as e:
                logger.warning(f"Network error: {e} - Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Max retries exceeded due to network error: {e}")

        return None

    def _log_success(self, result: Dict):
        """Helper to log token usage and reasoning content preview."""
        try:
            usage = result.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            
            message = result['choices'][0]['message']
            content = message.get('content', '') or ""
            reasoning = message.get('reasoning_content', '') or ""
            
            logger.info(f"API Success: {total_tokens} tokens used.")
            
            if reasoning:
                logger.debug(f"  Reasoning ({len(reasoning)} chars): {reasoning[:200]}...")
            
            if len(content) > 500:
                logger.debug(f"  Content ({len(content)} chars): {content[:200]}...{content[-100:]}")
            else:
                logger.debug(f"  Content: {content}")
                
        except Exception:
            # Don't let logging crash the flow
            pass
