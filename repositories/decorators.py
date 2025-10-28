"""Decorators for database operation retry logic."""
import time
import asyncio
from functools import wraps
from typing import Callable, Any
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure


def mongo_retry(max_retries: int = 3, delay: int = 1):
    """Decorator to retry MongoDB operations on connection failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds (will use exponential backoff)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @mongo_retry(max_retries=3, delay=1)
        def save_document(self, doc):
            return self.collection.insert_one(doc)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        print(f"MongoDB operation failed after {max_retries} attempts: {func.__name__}")
                        raise
                    
                    # Calculate exponential backoff delay
                    wait_time = delay * (2 ** attempt)
                    print(f"MongoDB operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                except Exception as e:
                    # For non-connection errors, don't retry
                    print(f"MongoDB operation '{func.__name__}' failed with non-retryable error: {e}")
                    raise
            
            # This should never be reached, but just in case
            return func(*args, **kwargs)
        return wrapper
    return decorator


def async_mongo_retry(max_retries: int = 3, delay: int = 1):
    """Decorator to retry async MongoDB operations on connection failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds (will use exponential backoff)
        
    Returns:
        Decorated async function with retry logic
        
    Example:
        @async_mongo_retry(max_retries=3, delay=1)
        async def save_document_async(self, doc):
            return await self.collection.insert_one(doc)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        print(f"Async MongoDB operation failed after {max_retries} attempts: {func.__name__}")
                        raise
                    
                    # Calculate exponential backoff delay
                    wait_time = delay * (2 ** attempt)
                    print(f"Async MongoDB operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # For non-connection errors, don't retry
                    print(f"Async MongoDB operation '{func.__name__}' failed with non-retryable error: {e}")
                    raise
            
            # This should never be reached, but just in case
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def redis_retry(max_retries: int = 3, delay: int = 1):
    """Decorator to retry Redis operations on connection failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds (will use exponential backoff)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            import redis.exceptions
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                    if attempt == max_retries - 1:
                        print(f"Redis operation failed after {max_retries} attempts: {func.__name__}")
                        raise
                    
                    wait_time = delay * (2 ** attempt)
                    print(f"Redis operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    print(f"Redis operation '{func.__name__}' failed with non-retryable error: {e}")
                    raise
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

