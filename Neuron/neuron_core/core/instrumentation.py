"""
instrumentation.py - OpenTelemetry Tracing for neuron_core

Provides Cloud Trace observability for the agent framework.
Falls back to console output when cloud credentials aren't available.
"""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

logger = logging.getLogger(__name__)


def setup_tracer(service_name: str = "neuron-core") -> trace.Tracer:
    """
    Initialize and configure OpenTelemetry tracing.
    
    Attempts to use Google Cloud Trace for production environments.
    Falls back to console output for local development.
    
    Args:
        service_name: Name of the service for trace identification
        
    Returns:
        Configured tracer object
    """
    # Create a resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Initialize the tracer provider
    provider = TracerProvider(resource=resource)
    
    # Try to use Cloud Trace exporter (production)
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        
        cloud_exporter = CloudTraceSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
        logger.info(f"Tracer initialized with Cloud Trace exporter for '{service_name}'")
        
    except Exception as e:
        # Fall back to console exporter (development)
        logger.warning(
            f"Cloud Trace exporter not available ({e}). "
            "Using console exporter for local development."
        )
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info(f"Tracer initialized with Console exporter for '{service_name}'")
    
    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    
    # Return a tracer for this service
    return trace.get_tracer(service_name)


def get_tracer(name: str = "neuron-core") -> trace.Tracer:
    """
    Get a tracer with the specified name.
    
    Uses the global tracer provider. Call setup_tracer() first
    to configure the provider.
    
    Args:
        name: Name for the tracer
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def traced(span_name: Optional[str] = None):
    """
    Decorator to automatically trace a function.
    
    Usage:
        @traced("process_message")
        async def process_message(self, message):
            ...
    
    Args:
        span_name: Name for the span (defaults to function name)
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on async or sync
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
