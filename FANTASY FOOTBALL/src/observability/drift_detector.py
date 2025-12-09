"""
Cultural Drift Detection

Alerts when regional voice parameters become unstable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RegionalProfile:
    """Baseline profile for a region"""
    region: str
    avg_tempo: float = 1.0
    avg_interruption_rate: float = 0.3
    expected_phrases: List[str] = field(default_factory=list)
    sentiment_range: tuple = (0.3, 0.7)


@dataclass
class DriftSample:
    """A single sample for drift detection"""
    region: str
    tempo: float
    interruption_rate: float
    sentiment: float
    phrase_match_rate: float
    timestamp: float = field(default_factory=time.time)


class CulturalDriftDetector:
    """
    Detects when regional personalities drift from baseline.
    
    Monitors:
    - Tempo consistency
    - Interruption rate
    - Sentiment range
    - Phrase usage
    
    Usage:
        detector = CulturalDriftDetector()
        
        # Set baselines
        detector.set_baseline("kansas_city", tempo=1.2, interruption_rate=0.4)
        
        # Record samples
        detector.record_sample("kansas_city", tempo=1.1, sentiment=0.6)
        
        # Check for drift
        alerts = detector.check_drift()
    """
    
    # Drift thresholds
    TEMPO_DRIFT_THRESHOLD = 0.3  # 30% deviation
    INTERRUPTION_DRIFT_THRESHOLD = 0.4
    SENTIMENT_DRIFT_THRESHOLD = 0.25
    PHRASE_MATCH_MIN = 0.5
    
    def __init__(
        self,
        on_drift_detected: Optional[Callable[[str, str, float], None]] = None,
        window_size: int = 100
    ):
        self.on_drift_detected = on_drift_detected
        self.window_size = window_size
        
        self._baselines: Dict[str, RegionalProfile] = {}
        self._samples: Dict[str, deque] = {}
        self._alerts: List[dict] = []
    
    def set_baseline(
        self,
        region: str,
        tempo: float = 1.0,
        interruption_rate: float = 0.3,
        expected_phrases: List[str] = None,
        sentiment_range: tuple = (0.3, 0.7)
    ):
        """Set baseline for a region"""
        self._baselines[region] = RegionalProfile(
            region=region,
            avg_tempo=tempo,
            avg_interruption_rate=interruption_rate,
            expected_phrases=expected_phrases or [],
            sentiment_range=sentiment_range
        )
        self._samples[region] = deque(maxlen=self.window_size)
    
    def record_sample(
        self,
        region: str,
        tempo: float = 1.0,
        interruption_rate: float = 0.0,
        sentiment: float = 0.5,
        phrase_match_rate: float = 1.0
    ):
        """Record a sample for drift analysis"""
        if region not in self._samples:
            self._samples[region] = deque(maxlen=self.window_size)
        
        sample = DriftSample(
            region=region,
            tempo=tempo,
            interruption_rate=interruption_rate,
            sentiment=sentiment,
            phrase_match_rate=phrase_match_rate
        )
        self._samples[region].append(sample)
    
    def check_drift(self, region: str = None) -> List[dict]:
        """Check for drift in one or all regions"""
        regions = [region] if region else list(self._baselines.keys())
        alerts = []
        
        for r in regions:
            if r not in self._baselines or r not in self._samples:
                continue
            
            if len(self._samples[r]) < 10:  # Need minimum samples
                continue
            
            baseline = self._baselines[r]
            samples = list(self._samples[r])
            
            # Check tempo drift
            tempo_drift = self._check_tempo_drift(baseline, samples)
            if tempo_drift:
                alerts.append(tempo_drift)
            
            # Check interruption drift
            int_drift = self._check_interruption_drift(baseline, samples)
            if int_drift:
                alerts.append(int_drift)
            
            # Check sentiment drift
            sent_drift = self._check_sentiment_drift(baseline, samples)
            if sent_drift:
                alerts.append(sent_drift)
            
            # Check phrase usage
            phrase_drift = self._check_phrase_drift(samples)
            if phrase_drift:
                alerts.append(phrase_drift)
        
        return alerts
    
    def _check_tempo_drift(self, baseline: RegionalProfile, samples: List[DriftSample]) -> Optional[dict]:
        """Check for tempo drift"""
        tempos = [s.tempo for s in samples]
        avg_tempo = statistics.mean(tempos)
        
        deviation = abs(avg_tempo - baseline.avg_tempo) / baseline.avg_tempo
        
        if deviation > self.TEMPO_DRIFT_THRESHOLD:
            return self._create_alert(
                baseline.region,
                "tempo",
                deviation,
                f"Tempo drift: {avg_tempo:.2f} vs baseline {baseline.avg_tempo:.2f}"
            )
        return None
    
    def _check_interruption_drift(self, baseline: RegionalProfile, samples: List[DriftSample]) -> Optional[dict]:
        """Check for interruption rate drift"""
        rates = [s.interruption_rate for s in samples]
        avg_rate = statistics.mean(rates)
        
        deviation = abs(avg_rate - baseline.avg_interruption_rate)
        
        if deviation > self.INTERRUPTION_DRIFT_THRESHOLD:
            return self._create_alert(
                baseline.region,
                "interruption_rate",
                deviation,
                f"Interruption drift: {avg_rate:.2f} vs baseline {baseline.avg_interruption_rate:.2f}"
            )
        return None
    
    def _check_sentiment_drift(self, baseline: RegionalProfile, samples: List[DriftSample]) -> Optional[dict]:
        """Check for sentiment drift"""
        sentiments = [s.sentiment for s in samples]
        avg_sentiment = statistics.mean(sentiments)
        
        low, high = baseline.sentiment_range
        
        if avg_sentiment < low - self.SENTIMENT_DRIFT_THRESHOLD:
            return self._create_alert(
                baseline.region,
                "sentiment_low",
                low - avg_sentiment,
                f"Sentiment too low: {avg_sentiment:.2f} vs range {low}-{high}"
            )
        
        if avg_sentiment > high + self.SENTIMENT_DRIFT_THRESHOLD:
            return self._create_alert(
                baseline.region,
                "sentiment_high",
                avg_sentiment - high,
                f"Sentiment too high: {avg_sentiment:.2f} vs range {low}-{high}"
            )
        return None
    
    def _check_phrase_drift(self, samples: List[DriftSample]) -> Optional[dict]:
        """Check for phrase usage drift"""
        if not samples:
            return None
        
        rates = [s.phrase_match_rate for s in samples]
        avg_rate = statistics.mean(rates)
        
        if avg_rate < self.PHRASE_MATCH_MIN:
            return self._create_alert(
                samples[0].region,
                "phrase_usage",
                self.PHRASE_MATCH_MIN - avg_rate,
                f"Expected phrases missing: {avg_rate:.1%} match rate"
            )
        return None
    
    def _create_alert(self, region: str, drift_type: str, deviation: float, message: str) -> dict:
        """Create a drift alert"""
        alert = {
            "region": region,
            "type": drift_type,
            "deviation": round(deviation, 3),
            "message": message,
            "timestamp": time.time()
        }
        
        self._alerts.append(alert)
        logger.warning(f"Cultural drift detected: {message}")
        
        if self.on_drift_detected:
            self.on_drift_detected(region, drift_type, deviation)
        
        return alert
    
    def get_status(self, region: str = None) -> dict:
        """Get drift status"""
        regions = [region] if region else list(self._baselines.keys())
        
        status = {}
        for r in regions:
            if r not in self._samples:
                continue
            
            samples = list(self._samples[r])
            if not samples:
                status[r] = {"status": "no_data"}
                continue
            
            baseline = self._baselines.get(r)
            
            tempos = [s.tempo for s in samples]
            sentiments = [s.sentiment for s in samples]
            
            status[r] = {
                "status": "healthy",
                "sample_count": len(samples),
                "current_tempo": round(statistics.mean(tempos), 2),
                "baseline_tempo": baseline.avg_tempo if baseline else None,
                "current_sentiment": round(statistics.mean(sentiments), 2),
                "recent_alerts": len([a for a in self._alerts if a["region"] == r])
            }
        
        return status
    
    def get_alerts(self, since: float = None) -> List[dict]:
        """Get recent alerts"""
        if since:
            return [a for a in self._alerts if a["timestamp"] >= since]
        return self._alerts[-50:]  # Last 50


# Singleton
_detector: Optional[CulturalDriftDetector] = None

def get_drift_detector() -> CulturalDriftDetector:
    global _detector
    if _detector is None:
        _detector = CulturalDriftDetector()
    return _detector
