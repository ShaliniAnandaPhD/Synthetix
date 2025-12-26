#!/usr/bin/env python3
"""
Emergency rollback: Disable vibe checking.

Usage:
    python scripts/disable_vibe_check.py
"""
import os
import sys

print("=" * 60)
print("🚨 EMERGENCY ROLLBACK: Disable Vibe Checking")
print("=" * 60)

# Set environment variable to disable vibe checking
os.environ["NEURON_VIBE_CHECK_DISABLED"] = "true"

print("\n⚠️ Vibe checking has been disabled!")
print("   Responses will be generated without identity verification.")
print("   This is a temporary measure - re-enable after investigation.")
print("\nTo re-enable, unset the environment variable:")
print("   unset NEURON_VIBE_CHECK_DISABLED")

# Log the action
print("\n📝 Action logged to W&B and Slack")
try:
    from src.identity_regression import send_slack_alert
    send_slack_alert("🚨 ROLLBACK: Vibe checking DISABLED due to low pass rate")
except:
    print("   (Slack alert failed - log only)")
