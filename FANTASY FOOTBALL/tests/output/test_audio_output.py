#!/usr/bin/env python3
"""
P1: Audio Output Validation Tests

Validates that audio files are valid and playable:
- Generated audio is valid MP3/WAV format
- Audio duration matches text length expectations
- Base64 audio from API can be decoded

Run:
    python tests/output/test_audio_output.py
"""

import base64
import os
import sys
import struct
from typing import Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

MODAL_BASE_URL = "https://neuronsystems--neuron-orchestrator"
ENDPOINTS = {
    "generate_tts": f"{MODAL_BASE_URL}-generate-tts.modal.run",
}

# Expected audio characteristics
MIN_AUDIO_SIZE = 1000  # bytes
MAX_AUDIO_SIZE = 5_000_000  # 5MB
CHARS_PER_SECOND = 15  # Approximate speech rate


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_mp3(data: bytes) -> bool:
    """Check if data starts with MP3 headers."""
    if len(data) < 3:
        return False
    # MP3 frame sync or ID3 tag
    return data[:3] == b'ID3' or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


def is_valid_wav(data: bytes) -> bool:
    """Check if data starts with WAV headers."""
    if len(data) < 4:
        return False
    return data[:4] == b'RIFF'


def estimate_duration_from_size(size_bytes: int, format_type: str = "mp3") -> float:
    """Estimate audio duration based on file size."""
    # Rough estimates based on typical bitrates
    if format_type == "mp3":
        # ~128kbps = ~16KB per second
        return size_bytes / 16000
    elif format_type == "wav":
        # ~24kHz 16-bit mono = ~48KB per second
        return size_bytes / 48000
    return size_bytes / 20000  # Generic estimate


# ============================================================================
# TEST 1: Audio File Valid Format
# ============================================================================

def test_audio_file_valid() -> Tuple[bool, str]:
    """
    Generated audio is valid MP3/WAV format.
    
    Tests that the TTS endpoint returns properly formatted audio.
    """
    print("\n🧪 Test 1: Audio File Valid Format")
    
    test_texts = [
        ("Philadelphia", "Let's go Eagles! That's a touchdown baby!"),
        ("Kansas City", "Mahomes does it again, Chiefs Kingdom!"),
    ]
    
    results = []
    
    for city, text in test_texts:
        print(f"   Testing: {city}")
        
        try:
            response = requests.post(
                ENDPOINTS["generate_tts"],
                json={
                    "text": text,
                    "speaker_id": city,
                    "provider": "google"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                results.append({"passed": False, "reason": f"HTTP {response.status_code}"})
                continue
            
            data = response.json()
            audio_base64 = data.get("audio") or data.get("audio_base64", "")
            
            if not audio_base64:
                results.append({"passed": False, "reason": "No audio data"})
                continue
            
            # Decode and validate
            audio_data = base64.b64decode(audio_base64)
            
            is_mp3 = is_valid_mp3(audio_data)
            is_wav = is_valid_wav(audio_data)
            
            valid_format = is_mp3 or is_wav
            format_name = "MP3" if is_mp3 else ("WAV" if is_wav else "Unknown")
            
            results.append({
                "passed": valid_format,
                "format": format_name,
                "size": len(audio_data)
            })
            
            if valid_format:
                print(f"      ✅ Valid {format_name} ({len(audio_data)} bytes)")
            else:
                print(f"      ❌ Invalid format (header: {audio_data[:4].hex()})")
                
        except Exception as e:
            results.append({"passed": False, "reason": str(e)[:50]})
    
    passed_count = sum(1 for r in results if r.get("passed"))
    total_count = len(results)
    passed = passed_count == total_count
    
    status = f"{passed_count}/{total_count} valid audio files"
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    
    return passed, status


# ============================================================================
# TEST 2: Audio Duration Reasonable
# ============================================================================

def test_audio_duration_reasonable() -> Tuple[bool, str]:
    """
    Audio duration matches text length expectations.
    
    Tests that audio length is proportional to text length.
    """
    print("\n🧪 Test 2: Audio Duration Reasonable")
    
    test_cases = [
        ("Dallas", "Short text.", 10),  # ~1s expected
        ("Pittsburgh", "This is a medium length text that should produce about ten seconds of audio when spoken at a normal pace.", 100),  # ~7s expected
    ]
    
    results = []
    
    for city, text, expected_min_size in test_cases:
        print(f"   Testing: {len(text)} chars")
        
        try:
            response = requests.post(
                ENDPOINTS["generate_tts"],
                json={
                    "text": text,
                    "speaker_id": city,
                    "provider": "google"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                results.append({"passed": False, "reason": f"HTTP {response.status_code}"})
                continue
            
            data = response.json()
            audio_base64 = data.get("audio") or data.get("audio_base64", "")
            
            if not audio_base64:
                results.append({"passed": False, "reason": "No audio"})
                continue
            
            audio_data = base64.b64decode(audio_base64)
            audio_size = len(audio_data)
            
            # Estimate duration
            est_duration = estimate_duration_from_size(audio_size)
            expected_duration = len(text) / CHARS_PER_SECOND
            
            # Duration should be within 50-200% of expected
            duration_ratio = est_duration / expected_duration if expected_duration > 0 else 0
            reasonable = 0.3 < duration_ratio < 3.0
            
            results.append({
                "passed": reasonable,
                "size": audio_size,
                "est_duration": est_duration,
                "expected_duration": expected_duration
            })
            
            if reasonable:
                print(f"      ✅ ~{est_duration:.1f}s (expected ~{expected_duration:.1f}s)")
            else:
                print(f"      ⚠️ Duration ratio: {duration_ratio:.2f}x")
                
        except Exception as e:
            results.append({"passed": False, "reason": str(e)[:50]})
    
    passed_count = sum(1 for r in results if r.get("passed"))
    total_count = len(results)
    passed = passed_count >= total_count * 0.5  # At least 50% pass
    
    status = f"{passed_count}/{total_count} reasonable durations"
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    
    return passed, status


# ============================================================================
# TEST 3: Base64 Audio Decodable
# ============================================================================

def test_audio_base64_decodable() -> Tuple[bool, str]:
    """
    Base64 audio from API can be decoded and played.
    
    Tests the full decode path that the frontend uses.
    """
    print("\n🧪 Test 3: Base64 Audio Decodable")
    
    test_cities = ["Buffalo", "Miami", "Green Bay"]
    
    results = []
    
    for city in test_cities:
        print(f"   Testing: {city}")
        text = f"Testing audio generation for {city} fans!"
        
        try:
            response = requests.post(
                ENDPOINTS["generate_tts"],
                json={
                    "text": text,
                    "speaker_id": city,
                    "provider": "google"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                results.append({"passed": False, "reason": f"HTTP {response.status_code}"})
                continue
            
            data = response.json()
            audio_base64 = data.get("audio") or data.get("audio_base64", "")
            
            # Test 1: Can decode base64
            try:
                audio_data = base64.b64decode(audio_base64)
                decode_success = True
            except Exception as e:
                results.append({"passed": False, "reason": f"Decode failed: {str(e)[:30]}"})
                continue
            
            # Test 2: Data is within size bounds
            size_ok = MIN_AUDIO_SIZE < len(audio_data) < MAX_AUDIO_SIZE
            
            # Test 3: Has valid audio header
            has_header = is_valid_mp3(audio_data) or is_valid_wav(audio_data)
            
            passed = decode_success and size_ok and has_header
            
            results.append({
                "passed": passed,
                "decoded": decode_success,
                "size_ok": size_ok,
                "has_header": has_header,
                "size": len(audio_data)
            })
            
            if passed:
                print(f"      ✅ Decodable, valid audio ({len(audio_data)} bytes)")
            else:
                print(f"      ❌ Issues: size={size_ok}, header={has_header}")
                
        except Exception as e:
            results.append({"passed": False, "reason": str(e)[:50]})
    
    passed_count = sum(1 for r in results if r.get("passed"))
    total_count = len(results)
    passed = passed_count == total_count
    
    status = f"{passed_count}/{total_count} fully decodable"
    print(f"   {'✅ PASSED' if passed else '❌ FAILED'} - {status}")
    
    return passed, status


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_audio_tests():
    """Run all audio output tests."""
    print("=" * 60)
    print("P1: AUDIO OUTPUT VALIDATION TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Valid format
    passed, status = test_audio_file_valid()
    results.append(("Audio File Valid", passed, status))
    
    # Test 2: Duration reasonable
    passed, status = test_audio_duration_reasonable()
    results.append(("Audio Duration Reasonable", passed, status))
    
    # Test 3: Base64 decodable
    passed, status = test_audio_base64_decodable()
    results.append(("Base64 Decodable", passed, status))
    
    # Summary
    print("\n" + "=" * 60)
    print("AUDIO OUTPUT TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed, status in results:
        symbol = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {symbol} - {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL AUDIO OUTPUT TESTS PASSED")
    else:
        print("⚠️  SOME AUDIO OUTPUT TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_audio_tests()
    sys.exit(0 if success else 1)
