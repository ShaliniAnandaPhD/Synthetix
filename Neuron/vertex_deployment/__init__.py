"""
vertex_deployment - Vertex AI Integration Package

Provides the adapter and utilities for deploying neuron_core
to Google Vertex AI Reasoning Engine.
"""

from .adapter import VertexNeuronAdapter

__all__ = ["VertexNeuronAdapter"]
__version__ = "1.0.0"
