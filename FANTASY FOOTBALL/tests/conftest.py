"""
Pytest Configuration and Fixtures

Shared fixtures for all test modules.
"""
import pytest
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# ASYNC SUPPORT
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def archetype_config():
    """Load archetype configuration once per session."""
    from src.identity_regression import load_archetype_config
    return load_archetype_config()


@pytest.fixture(scope="session")
def platinum_traces():
    """Load platinum traces once per session."""
    from src.identity_regression import load_platinum_traces
    return load_platinum_traces()


@pytest.fixture(scope="session")
def fallback_system():
    """Create fallback system once per session."""
    from src.identity_regression import PlatinumFallbackSystem
    return PlatinumFallbackSystem()


# =============================================================================
# TEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# =============================================================================
# MODAL FIXTURES
# =============================================================================

@pytest.fixture
def modal_debate_url():
    """Modal debate endpoint URL."""
    return "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"


@pytest.fixture
def modal_tts_url():
    """Modal TTS endpoint URL."""
    return "https://neuronsystems--neuron-orchestrator-generate-tts.modal.run"


@pytest.fixture
def modal_dashboard_url():
    """Modal dashboard API URL."""
    return "https://neuronsystems--neuron-orchestrator-dashboard-api.modal.run"
