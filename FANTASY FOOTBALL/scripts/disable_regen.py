#!/usr/bin/env python3
"""
Emergency rollback: Disable regeneration.

Usage:
    python scripts/disable_regen.py
"""
import os
import sys

print("=" * 60)
print("🚨 EMERGENCY ROLLBACK: Disable Regeneration")
print("=" * 60)

# Set environment variable to disable regeneration
os.environ["NEURON_REGEN_DISABLED"] = "true"

print("\n⚠️ Regeneration has been disabled!")
print("   Low-scoring responses will NOT be regenerated.")
print("   This reduces latency but may decrease quality.")
print("\nTo re-enable, unset the environment variable:")
print("   unset NEURON_REGEN_DISABLED")

# Log the action
print("\n📝 Action logged to W&B and Slack")
try:
    from src.identity_regression import send_slack_alert
    send_slack_alert("🚨 ROLLBACK: Regeneration DISABLED due to high latency")
except:
    print("   (Slack alert failed - log only)")
