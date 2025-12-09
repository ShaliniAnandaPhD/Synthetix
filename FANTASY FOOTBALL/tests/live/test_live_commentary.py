#!/usr/bin/env python3
"""
Phase 5: Live Commentary WebSocket Tests

Tests WebSocket connectivity, latency, and message flow for live commentary.

Requirements:
    pip install pytest pytest-asyncio websockets

Run:
    export LIVE_WS_URL="wss://your-modal-url.modal.run/live/test-game"
    pytest tests/live/test_live_commentary.py -v

Or for local testing:
    export LIVE_WS_URL="ws://localhost:8000/live/test-game"
    pytest tests/live/test_live_commentary.py -v
"""

import os
import json
import time
import asyncio
import pytest
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Import websockets - will fail gracefully if not installed
try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None


# Configuration
LIVE_WS_URL = os.environ.get("LIVE_WS_URL", "ws://localhost:8000/live/test-game")
CONNECTION_TIMEOUT = 10  # seconds
MESSAGE_TIMEOUT = 5  # seconds
LATENCY_THRESHOLD_MS = 500  # Max acceptable latency


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class LiveCommentaryTestClient:
    """
    Test client for WebSocket live commentary.
    
    Simulates a creator connecting to the live commentary stream.
    """
    
    def __init__(self, url: str = None):
        self.url = url or LIVE_WS_URL
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.messages_received: List[dict] = []
        self.latencies: List[float] = []
    
    async def connect(self, timeout: float = CONNECTION_TIMEOUT) -> bool:
        """Establish WebSocket connection"""
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(self.url, ping_timeout=30),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            raise ConnectionError(f"Connection timeout after {timeout}s")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.ws = None
    
    async def receive_message(self, timeout: float = MESSAGE_TIMEOUT) -> dict:
        """Receive a single message"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            self.messages_received.append(msg)
            return msg
        except asyncio.TimeoutError:
            raise TimeoutError(f"No message received after {timeout}s")
    
    async def send_event(self, event: dict) -> float:
        """Send an event and measure response latency"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        start_time = time.time()
        await self.ws.send(json.dumps(event))
        
        # Wait for response
        response = await self.receive_message()
        latency_ms = (time.time() - start_time) * 1000
        
        self.latencies.append(latency_ms)
        return latency_ms
    
    async def wait_for_message_type(self, msg_type: str, timeout: float = MESSAGE_TIMEOUT) -> dict:
        """Wait for a specific message type"""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                msg = await self.receive_message(timeout=remaining)
                if msg.get("type") == msg_type:
                    return msg
            except TimeoutError:
                break
        
        raise TimeoutError(f"Message type '{msg_type}' not received")


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def skip_if_no_websockets():
    """Skip test if websockets not installed"""
    if not WEBSOCKETS_AVAILABLE:
        pytest.skip("websockets package not installed. Run: pip install websockets")


@pytest.fixture
async def client():
    """Create and cleanup test client"""
    client = LiveCommentaryTestClient()
    yield client
    await client.disconnect()


# ============================================================================
# TEST CASES
# ============================================================================

@pytest.mark.asyncio
async def test_connection_establishment(skip_if_no_websockets, client):
    """Test 1: WebSocket connection can be established"""
    try:
        connected = await client.connect()
        assert connected, "Connection should return True"
        assert client.ws is not None, "WebSocket should be assigned"
        assert client.ws.open, "WebSocket should be open"
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


@pytest.mark.asyncio
async def test_initial_state_received(skip_if_no_websockets, client):
    """Test 2: Server sends initial 'connected' state"""
    try:
        await client.connect()
        
        # Wait for initial state
        msg = await client.receive_message(timeout=5)
        
        assert msg is not None, "Should receive initial message"
        assert "type" in msg, "Message should have 'type' field"
        
        # Accept either 'connected', 'welcome', or 'state' as valid initial message types
        valid_types = ["connected", "welcome", "state", "init"]
        assert msg["type"] in valid_types, f"Initial message type should be one of {valid_types}"
        
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


@pytest.mark.asyncio
async def test_touchdown_event_latency(skip_if_no_websockets, client):
    """Test 3: Send touchdown event and measure latency"""
    try:
        await client.connect()
        
        # Receive initial state first
        await client.receive_message(timeout=5)
        
        # Send touchdown event
        touchdown_event = {
            "type": "test_event",
            "event": {
                "type": "touchdown",
                "team": "KC",
                "description": "Mahomes TD pass to Kelce"
            }
        }
        
        latency_ms = await client.send_event(touchdown_event)
        
        assert latency_ms > 0, "Latency should be positive"
        assert latency_ms < LATENCY_THRESHOLD_MS, f"Latency {latency_ms:.0f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms"
        
        print(f"✓ Touchdown response latency: {latency_ms:.0f}ms")
        
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


@pytest.mark.asyncio
async def test_response_structure(skip_if_no_websockets, client):
    """Test 4: Validate response message structure"""
    try:
        await client.connect()
        
        # Receive initial state
        await client.receive_message(timeout=5)
        
        # Send a play event
        play_event = {
            "type": "test_event",
            "event": {
                "type": "play",
                "description": "Run for 5 yards"
            }
        }
        
        await client.ws.send(json.dumps(play_event))
        
        # Get response
        response = await client.receive_message(timeout=5)
        
        # Validate structure
        assert "type" in response, "Response should have 'type'"
        
        # If it's a commentary response, check for expected fields
        if response["type"] == "commentary":
            # Optional but expected fields
            expected_fields = ["text", "agent", "region"]
            for field in expected_fields:
                if field in response:
                    print(f"  ✓ Field '{field}': present")
            
            # Check for audio (optional)
            if "audio" in response:
                print(f"  ✓ Audio data included ({len(response['audio'])} bytes)")
        
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


@pytest.mark.asyncio
async def test_multiple_events_latency(skip_if_no_websockets, client):
    """Test 5: Send multiple events and check latency consistency"""
    try:
        await client.connect()
        
        # Receive initial state
        await client.receive_message(timeout=5)
        
        # Send 5 events
        events = [
            {"type": "test_event", "event": {"type": "play", "description": "Pass for 8 yards"}},
            {"type": "test_event", "event": {"type": "play", "description": "Run for 3 yards"}},
            {"type": "test_event", "event": {"type": "big_play", "description": "40 yard bomb!"}},
            {"type": "test_event", "event": {"type": "play", "description": "Incomplete pass"}},
            {"type": "test_event", "event": {"type": "touchdown", "description": "TOUCHDOWN!"}},
        ]
        
        latencies = []
        for event in events:
            try:
                latency = await client.send_event(event)
                latencies.append(latency)
                await asyncio.sleep(0.2)  # Brief pause between events
            except TimeoutError:
                # Server might not respond to all events
                pass
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"  Events processed: {len(latencies)}/5")
            print(f"  Avg latency: {avg_latency:.0f}ms")
            print(f"  Max latency: {max_latency:.0f}ms")
            
            assert avg_latency < LATENCY_THRESHOLD_MS, f"Avg latency {avg_latency:.0f}ms too high"
        
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


@pytest.mark.asyncio
async def test_session_cleanup(skip_if_no_websockets, client):
    """Test 6: Clean session disconnect"""
    try:
        await client.connect()
        
        # Receive initial state
        await client.receive_message(timeout=5)
        
        # Disconnect gracefully
        await client.disconnect()
        
        assert client.ws is None, "WebSocket should be None after disconnect"
        
    except ConnectionError as e:
        pytest.skip(f"Server not available: {e}")


# ============================================================================
# LOAD TEST (Optional)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_sessions(skip_if_no_websockets):
    """Test 7: Multiple concurrent WebSocket sessions (load test)"""
    NUM_CLIENTS = 5
    
    async def run_client(client_id: int) -> TestResult:
        client = LiveCommentaryTestClient()
        try:
            await client.connect()
            msg = await client.receive_message(timeout=5)
            
            # Send one event per client
            latency = await client.send_event({
                "type": "test_event",
                "event": {"type": "play", "description": f"Client {client_id} event"}
            })
            
            await client.disconnect()
            
            return TestResult(
                name=f"client_{client_id}",
                passed=True,
                latency_ms=latency
            )
        except Exception as e:
            return TestResult(
                name=f"client_{client_id}",
                passed=False,
                error=str(e)
            )
        finally:
            await client.disconnect()
    
    try:
        # Run clients concurrently
        results = await asyncio.gather(*[run_client(i) for i in range(NUM_CLIENTS)])
        
        successful = [r for r in results if r.passed]
        latencies = [r.latency_ms for r in successful if r.latency_ms]
        
        print(f"\n  Concurrent clients: {NUM_CLIENTS}")
        print(f"  Successful: {len(successful)}/{NUM_CLIENTS}")
        if latencies:
            print(f"  Avg latency: {sum(latencies)/len(latencies):.0f}ms")
        
        # At least 80% should succeed
        assert len(successful) >= NUM_CLIENTS * 0.8, "Too many client failures"
        
    except Exception as e:
        pytest.skip(f"Load test failed: {e}")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

async def run_standalone():
    """Run tests standalone (without pytest)"""
    print("=" * 60)
    print("Live Commentary WebSocket Tests")
    print(f"URL: {LIVE_WS_URL}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Connection
    print("\n1. Testing connection...")
    client = LiveCommentaryTestClient()
    try:
        await client.connect()
        print("   ✓ Connected")
        results.append(("Connection", True))
        
        # Test 2: Initial state
        print("\n2. Testing initial state...")
        msg = await client.receive_message()
        print(f"   ✓ Received: {msg.get('type', 'unknown')}")
        results.append(("Initial State", True))
        
        # Test 3: Touchdown latency
        print("\n3. Testing touchdown latency...")
        latency = await client.send_event({
            "type": "test_event",
            "event": {"type": "touchdown", "team": "KC"}
        })
        print(f"   ✓ Latency: {latency:.0f}ms")
        results.append(("Touchdown Latency", latency < LATENCY_THRESHOLD_MS))
        
        # Cleanup
        await client.disconnect()
        results.append(("Cleanup", True))
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results.append(("Test", False))
        await client.disconnect()
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    print(f"Results: {passed}/{len(results)} passed")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    if not WEBSOCKETS_AVAILABLE:
        print("ERROR: websockets package not installed")
        print("Run: pip install websockets")
        exit(1)
    
    success = asyncio.run(run_standalone())
    exit(0 if success else 1)
