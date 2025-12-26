"""
Real-time monitoring and alerting for game sessions.
"""
import os
import requests
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

if not SLACK_WEBHOOK_URL:
    logger.warning("SLACK_WEBHOOK_URL not set - alerts will be logged only")


def check_system_health(metrics: Dict) -> List[str]:
    """
    Run every 60s during game.
    Returns list of alerts if any thresholds are breached.
    
    Args:
        metrics: Dict with current metrics
    
    Returns:
        List of alert strings (empty if healthy)
    """
    alerts = []
    
    # Overall vibe check pass rate
    vibe_pass_rate = metrics.get("vibe_check_pass_rate", 1.0)
    if vibe_pass_rate < 0.75:
        alerts.append(f"⚠️ Vibe check degrading: {vibe_pass_rate:.0%} (threshold: 75%)")
    
    # Fallback rate
    fallback_rate = metrics.get("platinum_fallback_rate", 0.0)
    if fallback_rate > 0.10:
        alerts.append(f"⚠️ High fallback rate: {fallback_rate:.0%} (threshold: 10%)")
    
    # Regeneration rate
    regen_rate = metrics.get("regeneration_rate", 0.0)
    if regen_rate > 0.20:
        alerts.append(f"⚠️ High regeneration rate: {regen_rate:.0%} (threshold: 20%)")
    
    # Latency
    avg_latency = metrics.get("avg_latency", 0)
    if avg_latency > 60:
        alerts.append(f"⚠️ High latency: {avg_latency:.1f}s (threshold: 60s)")
    
    # Per-archetype drift
    archetypes = ["statistician", "hot_take_artist", "homer", "analyst", "historian", "neutral"]
    for arch in archetypes:
        pass_rate = metrics.get(f"vibe_check/{arch}/pass_rate", 1.0)
        if pass_rate < 0.70:
            alerts.append(f"🚨 {arch} COLLAPSING: {pass_rate:.0%} (threshold: 70%)")
    
    # Error rate
    error_rate = metrics.get("error_rate", 0.0)
    if error_rate > 0.05:
        alerts.append(f"🚨 High error rate: {error_rate:.0%} (threshold: 5%)")
    
    return alerts


def send_slack_alert(message: str) -> bool:
    """
    Send alert to Slack channel.
    
    Args:
        message: Alert message to send
    
    Returns:
        True if sent successfully, False otherwise
    """
    if not SLACK_WEBHOOK_URL:
        logger.warning(f"Would send Slack alert (webhook not configured): {message}")
        return False
    
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": message},
            timeout=5
        )
        response.raise_for_status()
        logger.info(f"Slack alert sent: {message[:100]}...")
        return True
    except requests.exceptions.Timeout:
        logger.error("Slack alert timeout")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def send_alerts_if_needed(metrics: Dict) -> int:
    """
    Check health and send alerts if needed.
    
    Args:
        metrics: Dict with current metrics
    
    Returns:
        Number of alerts sent
    """
    alerts = check_system_health(metrics)
    
    if not alerts:
        return 0
    
    message = "🏈 NEURON GAME ALERT\n" + "\n".join(alerts)
    send_slack_alert(message)
    logger.warning(message)
    
    return len(alerts)


def format_metrics_summary(metrics: Dict) -> str:
    """
    Format metrics for logging/display.
    """
    return f"""
📊 Current Metrics:
- Vibe Check Pass Rate: {metrics.get('vibe_check_pass_rate', 0):.0%}
- Regeneration Rate: {metrics.get('regeneration_rate', 0):.0%}
- Fallback Rate: {metrics.get('platinum_fallback_rate', 0):.0%}
- Avg Latency: {metrics.get('avg_latency', 0):.1f}s
- Total Requests: {metrics.get('total_requests', 0)}
"""
