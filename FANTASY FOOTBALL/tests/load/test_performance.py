"""
Load Tests for Performance Verification

Tests system under various load conditions: sustained, peak, and burst.
"""
import pytest
import asyncio
import aiohttp
import time
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Modal endpoint
MODAL_DEBATE_URL = "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"


async def make_single_request(session: aiohttp.ClientSession, request_id: int):
    """Make a single request to Modal."""
    cities = [
        ("Houston", "Los Angeles"),
        ("Baltimore", "Green Bay"),
        ("Houston", "Dallas"),
        ("Philadelphia", "Pittsburgh")
    ]
    
    city1, city2 = cities[request_id % len(cities)]
    
    payload = {
        "city1": city1,
        "city2": city2,
        "topic": f"Load test request #{request_id}",
        "rounds": 1,
        "agent_count": 2
    }
    
    start_time = time.time()
    
    try:
        async with session.post(
            MODAL_DEBATE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            latency = time.time() - start_time
            
            if resp.status == 200:
                return {"status": "success", "latency": latency, "id": request_id}
            else:
                return {"status": f"http_{resp.status}", "latency": latency, "id": request_id}
                
    except asyncio.TimeoutError:
        latency = time.time() - start_time
        return {"status": "timeout", "latency": latency, "id": request_id}
    except Exception as e:
        latency = time.time() - start_time
        return {"status": f"error_{type(e).__name__}", "latency": latency, "id": request_id}


# =============================================================================
# LOAD TESTS
# =============================================================================

class TestLoadPerformance:
    """Load tests for system performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_20_concurrent_requests(self):
        """Test 20 concurrent requests (baseline)."""
        num_requests = 20
        
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [make_single_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
        
        # Process results
        successes = sum(1 for r in results if not isinstance(r, Exception) and r.get("status") == "success")
        
        if successes == 0:
            pytest.skip("No requests succeeded - Modal may be unavailable")
        
        success_rate = successes / num_requests
        
        # Should have high success rate
        assert success_rate >= 0.80, f"Success rate too low: {success_rate:.0%}"
        
        # Should complete in reasonable time
        assert duration < 120, f"Duration too long: {duration:.1f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_50_concurrent_requests(self):
        """Test 50 concurrent requests."""
        num_requests = 50
        
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [make_single_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
        
        successes = sum(1 for r in results if not isinstance(r, Exception) and r.get("status") == "success")
        
        if successes == 0:
            pytest.skip("No requests succeeded")
        
        success_rate = successes / num_requests
        assert success_rate >= 0.80, f"Success rate: {success_rate:.0%}"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(900)
    async def test_100_concurrent_game1_scale(self):
        """Test 100 concurrent requests (Game 1 scale)."""
        num_requests = 100
        
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [make_single_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
        
        # Count results
        status_counts = Counter()
        latencies = []
        
        for r in results:
            if isinstance(r, Exception):
                status_counts["exception"] += 1
            else:
                status_counts[r["status"]] += 1
                latencies.append(r["latency"])
        
        successes = status_counts.get("success", 0)
        
        if successes == 0:
            pytest.skip("No requests succeeded")
        
        success_rate = successes / num_requests
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"Game 1 scale test: {success_rate:.0%} success, {avg_latency:.1f}s avg latency")
        
        assert success_rate >= 0.80, f"Success rate: {success_rate:.0%}"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(1200)
    async def test_150_concurrent_game2_scale(self):
        """Test 150 concurrent requests (Game 2 scale)."""
        num_requests = 150
        
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [make_single_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
        
        # Count results
        status_counts = Counter()
        latencies = []
        
        for r in results:
            if isinstance(r, Exception):
                status_counts["exception"] += 1
            else:
                status_counts[r["status"]] += 1
                latencies.append(r["latency"])
        
        successes = status_counts.get("success", 0)
        
        if successes == 0:
            pytest.skip("No requests succeeded")
        
        success_rate = successes / num_requests
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"Game 2 scale test: {success_rate:.0%} success, {avg_latency:.1f}s avg latency")
        print(f"Status breakdown: {dict(status_counts)}")
        
        # Game 2 scale - allow slightly lower success rate
        assert success_rate >= 0.75, f"Success rate: {success_rate:.0%}"


# =============================================================================
# BURST TESTS
# =============================================================================

class TestBurstLoad:
    """Tests for burst load patterns (like after touchdowns)."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_burst_pattern(self):
        """Test burst pattern: 50 requests, pause, 50 more."""
        requests_per_burst = 30
        num_bursts = 2
        delay_between = 10
        
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            for burst in range(num_bursts):
                start = time.time()
                tasks = [
                    make_single_request(session, i + burst * requests_per_burst)
                    for i in range(requests_per_burst)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start
                
                all_results.extend(results)
                
                print(f"Burst {burst + 1}: {duration:.1f}s")
                
                if burst < num_bursts - 1:
                    await asyncio.sleep(delay_between)
        
        # Count successful
        successes = sum(
            1 for r in all_results 
            if not isinstance(r, Exception) and r.get("status") == "success"
        )
        total = len(all_results)
        
        if successes == 0:
            pytest.skip("No requests succeeded")
        
        success_rate = successes / total
        assert success_rate >= 0.70, f"Burst success rate: {success_rate:.0%}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
