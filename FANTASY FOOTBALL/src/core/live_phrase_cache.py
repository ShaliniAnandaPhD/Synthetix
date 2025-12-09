"""
Live Voice Phrase Cache

Pre-generates common reaction phrases before games for instant playback.
Solves the 150ms latency budget problem by caching audio.
"""

import asyncio
import hashlib
import random
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedReaction:
    """A pre-generated voice reaction"""
    text: str
    audio_base64: str
    emotion: str
    duration_ms: int
    agent_type: str
    region: str


class LivePhraseCache:
    """
    Pre-generated voice clips for instant reactions.
    Target: <50ms cached reaction delivery.
    
    Usage:
        cache = LivePhraseCache(voice_engine, redis_client)
        await cache.warm_for_game("KC_BUF", ["kansas_city", "buffalo"])
        
        # During live game:
        reaction = await cache.get_instant_reaction("kansas_city", "homer", "touchdown")
        if reaction:
            # Instant playback!
            send_audio(reaction.audio_base64)
    """
    
    # Common reaction templates by event type and agent
    REACTION_TEMPLATES = {
        "touchdown": {
            "homer": [
                "TOUCHDOWN! That's what I'm talking about!",
                "YES! Did you see that?!",
                "THERE IT IS! I told you!",
                "Let's GO! That's my guy!",
                "TOUCHDOWN! Oh baby!",
                "SIX POINTS! Let's gooo!",
            ],
            "analyst": [
                "And that's six. Perfect execution.",
                "Touchdown. The coverage completely broke down.",
                "That's the play we talked about. Wide open.",
                "Six points on the board. Great scheme.",
                "Touchdown. Exactly what we expected.",
            ],
            "contrarian": [
                "Okay, but let's see if they can do it again.",
                "One touchdown doesn't change my take.",
                "Sure, but the defense gave that away.",
                "I still have concerns despite the score.",
            ],
            "historian": [
                "That reminds me of the great plays from years past.",
                "We've seen this before. Classic execution.",
                "Historically, this team thrives in these moments.",
            ],
        },
        "turnover": {
            "homer": [
                "NO! Come ON!",
                "You've got to be kidding me!",
                "Unbelievable. Just unbelievable.",
                "That's a disaster right there!",
                "NO NO NO!",
            ],
            "analyst": [
                "That's a killer. Momentum shift.",
                "Ball security. We talked about this.",
                "That's going to hurt. Bad decision.",
                "Critical turnover. Game-changing.",
            ],
            "contrarian": [
                "I saw this coming.",
                "This is exactly what I was worried about.",
                "Told you so. Ball security issues.",
            ],
        },
        "big_play": {
            "homer": [
                "WHAT A PLAY! Did you see that?!",
                "OH! Big time play right there!",
                "NOW we're talking!",
                "That's what I'm talking about!",
            ],
            "analyst": [
                "Great scheme recognition there.",
                "He found the gap. Nice read.",
                "Big play. Took advantage of the coverage.",
                "Excellent execution on that one.",
            ],
        },
        "interception": {
            "homer": [
                "PICKED OFF! Yes!",
                "INTERCEPTION! Great play!",
                "He read that perfectly!",
            ],
            "analyst": [
                "Bad throw. Should have seen that coming.",
                "Interception. Forced it into coverage.",
                "Risky decision there. Paid the price.",
            ],
        },
        "injury": {
            "homer": [
                "Oh no. That doesn't look good.",
                "Please be okay. Please be okay.",
                "Hate to see that happen.",
            ],
            "analyst": [
                "Injury timeout. Let's hope it's not serious.",
                "That's concerning. Medical staff out there.",
                "Never want to see injuries. Hope he's alright.",
            ],
        },
        "penalty": {
            "homer": [
                "Come ON! That's a bad call!",
                "Are you kidding me, ref?!",
                "Terrible call right there.",
            ],
            "analyst": [
                "That's a fair call. Can't do that.",
                "Penalty will hurt them here.",
                "Discipline issue there.",
            ],
        },
    }
    
    # Regional voice IDs
    REGIONAL_VOICES = {
        "kansas_city": "21m00Tcm4TlvDq8ikWAM",
        "dallas": "JBFqnCBsd6RMkjVDRZzb",
        "buffalo": "VR6AewLTigWG4xSOukaG",
        "philadelphia": "pNInz6obpgDQGcFmaJgB",
        "new_york": "ErXwobaYiN019PkySvjV",
        "green_bay": "pNInz6obpgDQGcFmaJgB",
        "san_francisco": "ThT5KcBeYPX3keUQqHPh",
        "chicago": "VR6AewLTigWG4xSOukaG",
        "default": "onwK4e9ZLuTAKqWW03F9",
    }
    
    def __init__(self, voice_engine=None, redis_client=None):
        """
        Initialize phrase cache.
        
        Args:
            voice_engine: Voice synthesis engine (ElevenLabs)
            redis_client: Redis client for distributed cache
        """
        self.voice_engine = voice_engine
        self.redis = redis_client
        self._local_cache: Dict[str, CachedReaction] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
    
    async def warm_for_game(
        self, 
        game_id: str, 
        regions: List[str],
        agent_types: List[str] = None,
        event_types: List[str] = None
    ) -> dict:
        """
        Pre-generate all common reactions before game starts.
        Call this 30 minutes before kickoff.
        
        Returns:
            Dict with warming stats
        """
        if agent_types is None:
            agent_types = ["homer", "analyst", "contrarian"]
        
        if event_types is None:
            event_types = ["touchdown", "turnover", "big_play", "interception"]
        
        start_time = time.time()
        phrases_cached = 0
        errors = 0
        
        logger.info(f"Warming phrase cache for {game_id} with {len(regions)} regions...")
        
        for region in regions:
            voice_id = self._get_voice_id(region)
            
            for event_type in event_types:
                templates = self.REACTION_TEMPLATES.get(event_type, {})
                
                for agent_type in agent_types:
                    phrases = templates.get(agent_type, [])
                    
                    for phrase in phrases:
                        try:
                            cache_key = self._cache_key(region, agent_type, event_type, phrase)
                            
                            # Check if already cached
                            if await self._is_cached(cache_key):
                                continue
                            
                            # Generate audio
                            if self.voice_engine:
                                audio = await self._generate_audio(phrase, voice_id, event_type)
                            else:
                                # Mock audio for testing
                                audio = self._mock_audio(phrase)
                            
                            # Store in cache
                            reaction = CachedReaction(
                                text=phrase,
                                audio_base64=audio,
                                emotion=self._get_emotion(event_type),
                                duration_ms=len(phrase) * 60,  # ~60ms per character
                                agent_type=agent_type,
                                region=region
                            )
                            
                            await self._store_cached(cache_key, reaction)
                            phrases_cached += 1
                            
                        except Exception as e:
                            logger.error(f"Error caching phrase: {e}")
                            errors += 1
        
        duration = time.time() - start_time
        stats = {
            "game_id": game_id,
            "regions": regions,
            "phrases_cached": phrases_cached,
            "errors": errors,
            "duration_seconds": round(duration, 2)
        }
        
        logger.info(f"Cache warmed: {phrases_cached} phrases in {duration:.1f}s")
        return stats
    
    async def get_instant_reaction(
        self, 
        region: str, 
        agent_type: str, 
        event_type: str
    ) -> Optional[CachedReaction]:
        """
        Get a pre-cached reaction for instant playback.
        Target: <50ms retrieval time.
        
        Returns:
            CachedReaction or None if cache miss
        """
        # Get available phrases for this combo
        templates = self.REACTION_TEMPLATES.get(event_type, {})
        phrases = templates.get(agent_type, [])
        
        if not phrases:
            self._cache_stats["misses"] += 1
            return None
        
        # Try random phrases until we find one cached
        random.shuffle(phrases)
        
        for phrase in phrases:
            cache_key = self._cache_key(region, agent_type, event_type, phrase)
            reaction = await self._get_cached(cache_key)
            
            if reaction:
                self._cache_stats["hits"] += 1
                return reaction
        
        self._cache_stats["misses"] += 1
        return None
    
    def get_cache_stats(self) -> dict:
        """Get cache hit/miss statistics"""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate_percent": round(hit_rate, 1),
            "local_cache_size": len(self._local_cache)
        }
    
    def _cache_key(self, region: str, agent_type: str, event_type: str, phrase: str) -> str:
        """Generate cache key for a phrase"""
        phrase_hash = hashlib.md5(phrase.encode()).hexdigest()[:8]
        return f"live_phrase:{region}:{agent_type}:{event_type}:{phrase_hash}"
    
    def _get_voice_id(self, region: str) -> str:
        """Get voice ID for region"""
        return self.REGIONAL_VOICES.get(region, self.REGIONAL_VOICES["default"])
    
    def _get_emotion(self, event_type: str) -> str:
        """Map event type to emotion"""
        emotions = {
            "touchdown": "excited",
            "turnover": "frustrated",
            "big_play": "excited",
            "interception": "mixed",
            "injury": "concerned",
            "penalty": "frustrated",
        }
        return emotions.get(event_type, "neutral")
    
    async def _generate_audio(self, text: str, voice_id: str, event_type: str) -> str:
        """Generate audio using voice engine"""
        emotion = self._get_emotion(event_type)
        
        # Use voice engine's synthesis method
        audio = await self.voice_engine.synthesize(
            text=text,
            voice_id=voice_id,
            emotion=emotion
        )
        
        return audio
    
    def _mock_audio(self, text: str) -> str:
        """Generate mock audio for testing"""
        import base64
        mock_data = f"MOCK_AUDIO_{text[:20]}".encode()
        return base64.b64encode(mock_data).decode()
    
    async def _is_cached(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key in self._local_cache:
            return True
        if self.redis:
            return await self.redis.exists(key)
        return False
    
    async def _get_cached(self, key: str) -> Optional[CachedReaction]:
        """Get reaction from cache"""
        # Check local cache first
        if key in self._local_cache:
            return self._local_cache[key]
        
        # Check Redis
        if self.redis:
            data = await self.redis.get(key)
            if data:
                import json
                obj = json.loads(data)
                return CachedReaction(**obj)
        
        return None
    
    async def _store_cached(self, key: str, reaction: CachedReaction):
        """Store reaction in cache"""
        # Local cache
        self._local_cache[key] = reaction
        
        # Redis (with 4 hour TTL)
        if self.redis:
            import json
            data = json.dumps({
                "text": reaction.text,
                "audio_base64": reaction.audio_base64,
                "emotion": reaction.emotion,
                "duration_ms": reaction.duration_ms,
                "agent_type": reaction.agent_type,
                "region": reaction.region,
            })
            await self.redis.set(key, data, ex=14400)


# Singleton instance
_phrase_cache: Optional[LivePhraseCache] = None

def get_phrase_cache(voice_engine=None, redis_client=None) -> LivePhraseCache:
    """Get or create phrase cache singleton"""
    global _phrase_cache
    if _phrase_cache is None:
        _phrase_cache = LivePhraseCache(voice_engine, redis_client)
    return _phrase_cache


if __name__ == "__main__":
    # Test the cache
    async def test():
        cache = LivePhraseCache()
        
        # Warm cache
        stats = await cache.warm_for_game(
            "KC_BUF",
            ["kansas_city", "buffalo"],
            agent_types=["homer", "analyst"],
            event_types=["touchdown", "turnover"]
        )
        print(f"\nWarming stats: {stats}")
        
        # Get reactions
        reaction = await cache.get_instant_reaction("kansas_city", "homer", "touchdown")
        if reaction:
            print(f"\nReaction: {reaction.text}")
            print(f"Emotion: {reaction.emotion}")
        
        print(f"\nCache stats: {cache.get_cache_stats()}")
    
    asyncio.run(test())
