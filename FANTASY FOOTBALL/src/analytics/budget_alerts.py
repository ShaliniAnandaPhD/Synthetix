"""
Budget Alerts System

Daily/weekly spend alerts with auto-throttle.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BudgetConfig:
    """Budget configuration"""
    daily_limit_usd: float = 50.0
    weekly_limit_usd: float = 250.0
    monthly_limit_usd: float = 1000.0
    
    # Thresholds (percentage)
    warn_threshold: float = 0.75
    critical_threshold: float = 0.90
    
    # Auto-throttle
    throttle_at_limit: bool = True
    throttle_rate: float = 0.5  # Reduce to 50% capacity


@dataclass
class SpendRecord:
    """A spending record"""
    amount_usd: float
    category: str
    creator_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """A budget alert"""
    level: AlertLevel
    message: str
    current_spend: float
    limit: float
    period: str
    timestamp: float = field(default_factory=time.time)


class BudgetAlertSystem:
    """
    Track spending and send alerts when approaching limits.
    
    Features:
    - Daily, weekly, monthly limits
    - Warning and critical thresholds
    - Auto-throttle when limit reached
    - Email/webhook notifications
    
    Usage:
        alerts = BudgetAlertSystem()
        
        # Configure limits
        alerts.configure(daily_limit=50, weekly_limit=250)
        
        # Record spending
        alerts.record_spend(0.15, "llm", "creator123")
        
        # Check if throttled
        if alerts.is_throttled():
            # Reduce capacity
            pass
    """
    
    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        on_alert: Optional[Callable[[Alert], None]] = None
    ):
        self.config = config or BudgetConfig()
        self.on_alert = on_alert
        
        self._records: List[SpendRecord] = []
        self._alerts: List[Alert] = []
        self._is_throttled = False
        self._alerted_periods: set = set()  # Avoid duplicate alerts
    
    def configure(
        self,
        daily_limit: Optional[float] = None,
        weekly_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None
    ):
        """Update budget configuration"""
        if daily_limit is not None:
            self.config.daily_limit_usd = daily_limit
        if weekly_limit is not None:
            self.config.weekly_limit_usd = weekly_limit
        if monthly_limit is not None:
            self.config.monthly_limit_usd = monthly_limit
    
    def record_spend(
        self, 
        amount_usd: float, 
        category: str = "general",
        creator_id: str = ""
    ):
        """Record a spending event"""
        record = SpendRecord(
            amount_usd=amount_usd,
            category=category,
            creator_id=creator_id
        )
        self._records.append(record)
        
        # Check limits
        self._check_limits()
    
    def get_daily_spend(self) -> float:
        """Get total spend for today"""
        cutoff = self._start_of_day()
        return sum(r.amount_usd for r in self._records if r.timestamp >= cutoff)
    
    def get_weekly_spend(self) -> float:
        """Get total spend for this week"""
        cutoff = self._start_of_week()
        return sum(r.amount_usd for r in self._records if r.timestamp >= cutoff)
    
    def get_monthly_spend(self) -> float:
        """Get total spend for this month"""
        cutoff = self._start_of_month()
        return sum(r.amount_usd for r in self._records if r.timestamp >= cutoff)
    
    def is_throttled(self) -> bool:
        """Check if spending should be throttled"""
        return self._is_throttled
    
    def get_status(self) -> dict:
        """Get current budget status"""
        daily = self.get_daily_spend()
        weekly = self.get_weekly_spend()
        monthly = self.get_monthly_spend()
        
        return {
            "daily": {
                "spent_usd": round(daily, 2),
                "limit_usd": self.config.daily_limit_usd,
                "remaining_usd": round(max(0, self.config.daily_limit_usd - daily), 2),
                "percent": round(daily / self.config.daily_limit_usd * 100, 1)
            },
            "weekly": {
                "spent_usd": round(weekly, 2),
                "limit_usd": self.config.weekly_limit_usd,
                "remaining_usd": round(max(0, self.config.weekly_limit_usd - weekly), 2),
                "percent": round(weekly / self.config.weekly_limit_usd * 100, 1)
            },
            "monthly": {
                "spent_usd": round(monthly, 2),
                "limit_usd": self.config.monthly_limit_usd,
                "remaining_usd": round(max(0, self.config.monthly_limit_usd - monthly), 2),
                "percent": round(monthly / self.config.monthly_limit_usd * 100, 1)
            },
            "is_throttled": self._is_throttled,
            "recent_alerts": [
                {
                    "level": a.level.value,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in self._alerts[-5:]
            ]
        }
    
    def get_alerts(self, since: Optional[float] = None) -> List[dict]:
        """Get alerts, optionally filtered by time"""
        alerts = self._alerts
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return [
            {
                "level": a.level.value,
                "message": a.message,
                "current_spend": round(a.current_spend, 2),
                "limit": a.limit,
                "period": a.period,
                "timestamp": a.timestamp
            }
            for a in alerts
        ]
    
    def _check_limits(self):
        """Check all limits and trigger alerts"""
        checks = [
            ("daily", self.get_daily_spend(), self.config.daily_limit_usd),
            ("weekly", self.get_weekly_spend(), self.config.weekly_limit_usd),
            ("monthly", self.get_monthly_spend(), self.config.monthly_limit_usd),
        ]
        
        for period, spent, limit in checks:
            percent = spent / limit if limit > 0 else 0
            
            # Check for critical (exceeds limit)
            if percent >= 1.0:
                self._trigger_alert(
                    AlertLevel.CRITICAL,
                    f"{period.upper()} limit exceeded: ${spent:.2f} / ${limit:.2f}",
                    spent, limit, period
                )
                if self.config.throttle_at_limit:
                    self._is_throttled = True
            
            # Check for critical threshold
            elif percent >= self.config.critical_threshold:
                self._trigger_alert(
                    AlertLevel.CRITICAL,
                    f"{period.upper()} {int(percent*100)}% of limit: ${spent:.2f} / ${limit:.2f}",
                    spent, limit, f"{period}_critical"
                )
            
            # Check for warning threshold
            elif percent >= self.config.warn_threshold:
                self._trigger_alert(
                    AlertLevel.WARNING,
                    f"{period.upper()} {int(percent*100)}% of limit: ${spent:.2f} / ${limit:.2f}",
                    spent, limit, f"{period}_warning"
                )
    
    def _trigger_alert(
        self, 
        level: AlertLevel, 
        message: str,
        current: float,
        limit: float,
        period: str
    ):
        """Trigger an alert"""
        # Avoid duplicate alerts for same period
        alert_key = f"{period}_{datetime.now().strftime('%Y-%m-%d')}"
        if alert_key in self._alerted_periods:
            return
        
        self._alerted_periods.add(alert_key)
        
        alert = Alert(
            level=level,
            message=message,
            current_spend=current,
            limit=limit,
            period=period
        )
        self._alerts.append(alert)
        
        logger.log(
            logging.WARNING if level == AlertLevel.WARNING else logging.ERROR,
            message
        )
        
        if self.on_alert:
            self.on_alert(alert)
    
    def _start_of_day(self) -> float:
        """Get timestamp for start of today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return today.timestamp()
    
    def _start_of_week(self) -> float:
        """Get timestamp for start of this week (Monday)"""
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return monday.timestamp()
    
    def _start_of_month(self) -> float:
        """Get timestamp for start of this month"""
        first = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return first.timestamp()
    
    def reset_throttle(self):
        """Manually reset throttle (use with caution)"""
        self._is_throttled = False
        logger.info("Throttle manually reset")


# Singleton
_alert_system: Optional[BudgetAlertSystem] = None

def get_budget_alerts() -> BudgetAlertSystem:
    global _alert_system
    if _alert_system is None:
        _alert_system = BudgetAlertSystem()
    return _alert_system
