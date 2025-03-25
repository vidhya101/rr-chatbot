import os
import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from threading import Lock
from utils.retry import retry, RetryContext
from utils.circuit_breaker import CircuitBreaker
from utils.exceptions import (
    MonitoringError, AlertError, MetricsError,
    ThresholdError, ResourceCleanupError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    level: str
    message: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float
    context: Dict[str, Any]

@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    warning: float
    critical: float
    duration: int  # Duration in seconds for sustained threshold breach

class MonitoringService:
    """Service for system monitoring, metrics collection, and alerting"""
    
    def __init__(self, alert_history_size: int = 1000):
        """Initialize monitoring service"""
        # Metrics storage with timestamps
        self.metrics_history: Dict[str, deque] = {
            'cpu_usage': deque(maxlen=3600),  # 1 hour of per-second data
            'memory_usage': deque(maxlen=3600),
            'disk_usage': deque(maxlen=3600),
            'network_io': deque(maxlen=3600),
            'response_times': deque(maxlen=3600),
            'error_rates': deque(maxlen=3600),
            'request_rates': deque(maxlen=3600)
        }
        
        # Alert configuration
        self.thresholds = {
            'cpu_usage': MetricThreshold(warning=70.0, critical=90.0, duration=300),
            'memory_usage': MetricThreshold(warning=70.0, critical=90.0, duration=300),
            'disk_usage': MetricThreshold(warning=80.0, critical=95.0, duration=600),
            'error_rate': MetricThreshold(warning=5.0, critical=10.0, duration=300),
            'response_time': MetricThreshold(warning=1.0, critical=3.0, duration=300)
        }
        
        # Alert history
        self.alerts = deque(maxlen=alert_history_size)
        self.active_alerts: Dict[str, Alert] = {}
        
        # Thread safety
        self.metrics_lock = Lock()
        self.alerts_lock = Lock()
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0
        )
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        self.baseline_samples = 1000
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        try:
            metrics = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().timestamp()
            }
            
            # Collect network I/O stats
            net_io = psutil.net_io_counters()
            metrics['network_bytes_sent'] = net_io.bytes_sent
            metrics['network_bytes_recv'] = net_io.bytes_recv
            
            # Store metrics in history
            with self.metrics_lock:
                for key, value in metrics.items():
                    if key in self.metrics_history:
                        self.metrics_history[key].append((datetime.now(), value))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            raise MetricsError(f"Failed to collect system metrics: {str(e)}")
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def collect_application_metrics(self, app_metrics: Dict[str, Any]) -> None:
        """Collect application-specific metrics"""
        try:
            timestamp = datetime.now()
            
            with self.metrics_lock:
                # Store response times
                if 'response_time' in app_metrics:
                    self.metrics_history['response_times'].append(
                        (timestamp, app_metrics['response_time'])
                    )
                
                # Store error rates
                if 'error_count' in app_metrics:
                    self.metrics_history['error_rates'].append(
                        (timestamp, app_metrics['error_count'])
                    )
                
                # Store request rates
                if 'request_count' in app_metrics:
                    self.metrics_history['request_rates'].append(
                        (timestamp, app_metrics['request_count'])
                    )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
            raise MetricsError(f"Failed to collect application metrics: {str(e)}")
    
    def check_thresholds(self) -> List[Alert]:
        """Check all metrics against thresholds"""
        try:
            new_alerts = []
            
            for metric, threshold in self.thresholds.items():
                if metric not in self.metrics_history:
                    continue
                
                # Get recent values for the metric
                recent_values = list(self.metrics_history[metric])[-threshold.duration:]
                if not recent_values:
                    continue
                
                # Calculate average over the duration
                avg_value = sum(v[1] for v in recent_values) / len(recent_values)
                
                # Check thresholds
                if avg_value >= threshold.critical:
                    alert = self._create_alert(
                        'critical',
                        f"{metric} critical threshold breached: {avg_value:.1f}%",
                        metric,
                        avg_value,
                        threshold.critical
                    )
                    new_alerts.append(alert)
                elif avg_value >= threshold.warning:
                    alert = self._create_alert(
                        'warning',
                        f"{metric} warning threshold breached: {avg_value:.1f}%",
                        metric,
                        avg_value,
                        threshold.warning
                    )
                    new_alerts.append(alert)
            
            # Update active alerts
            with self.alerts_lock:
                for alert in new_alerts:
                    self.active_alerts[alert.metric] = alert
                    self.alerts.append(alert)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")
            raise ThresholdError(f"Failed to check thresholds: {str(e)}")
    
    def get_metrics_summary(self, duration: int = 3600) -> Dict[str, Any]:
        """Get summary of all metrics for the specified duration"""
        try:
            summary = {}
            current_time = datetime.now()
            start_time = current_time - timedelta(seconds=duration)
            
            with self.metrics_lock:
                for metric, history in self.metrics_history.items():
                    # Filter values within the duration
                    values = [
                        v[1] for v in history
                        if v[0] >= start_time
                    ]
                    
                    if values:
                        summary[metric] = {
                            'current': values[-1],
                            'min': min(values),
                            'max': max(values),
                            'avg': sum(values) / len(values)
                        }
                        
                        # Add baseline comparison if available
                        if metric in self.baselines:
                            baseline = self.baselines[metric]
                            current = values[-1]
                            summary[metric]['baseline_diff'] = (
                                ((current - baseline) / baseline) * 100
                                if baseline != 0 else 0
                            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            raise MetricsError(f"Failed to get metrics summary: {str(e)}")
    
    def update_baseline(self, metric: str) -> None:
        """Update baseline for a metric using recent history"""
        try:
            with self.metrics_lock:
                if metric not in self.metrics_history:
                    raise MetricsError(f"Unknown metric: {metric}")
                
                values = list(self.metrics_history[metric])[-self.baseline_samples:]
                if values:
                    self.baselines[metric] = sum(v[1] for v in values) / len(values)
                
        except Exception as e:
            logger.error(f"Error updating baseline: {str(e)}")
            raise MetricsError(f"Failed to update baseline: {str(e)}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        with self.alerts_lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self) -> List[Alert]:
        """Get alert history"""
        with self.alerts_lock:
            return list(self.alerts)
    
    def _create_alert(
        self, level: str, message: str, metric: str,
        value: float, threshold: float
    ) -> Alert:
        """Create a new alert"""
        return Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            metric=metric,
            value=value,
            threshold=threshold,
            context=self.get_metrics_summary(duration=300)  # Last 5 minutes
        )
    
    def analyze_performance_trend(
        self, metric: str, duration: int = 3600
    ) -> Dict[str, Any]:
        """Analyze performance trend for a metric"""
        try:
            with self.metrics_lock:
                if metric not in self.metrics_history:
                    raise MetricsError(f"Unknown metric: {metric}")
                
                # Get values within duration
                current_time = datetime.now()
                start_time = current_time - timedelta(seconds=duration)
                values = [
                    (v[0].timestamp(), v[1])
                    for v in self.metrics_history[metric]
                    if v[0] >= start_time
                ]
                
                if not values:
                    return {
                        'trend': 'no_data',
                        'slope': 0.0,
                        'volatility': 0.0
                    }
                
                # Calculate trend
                times = [v[0] for v in values]
                metrics = [v[1] for v in values]
                
                # Simple linear regression
                n = len(values)
                sum_x = sum(times)
                sum_y = sum(metrics)
                sum_xy = sum(x * y for x, y in zip(times, metrics))
                sum_xx = sum(x * x for x in times)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                
                # Calculate volatility (standard deviation)
                mean = sum_y / n
                variance = sum((y - mean) ** 2 for y in metrics) / n
                volatility = variance ** 0.5
                
                return {
                    'trend': 'increasing' if slope > 0 else 'decreasing',
                    'slope': slope,
                    'volatility': volatility
                }
                
        except Exception as e:
            logger.error(f"Error analyzing performance trend: {str(e)}")
            raise MetricsError(f"Failed to analyze performance trend: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup_resources()
    
    def _cleanup_resources(self) -> None:
        """Clean up monitoring resources"""
        try:
            # Clear metrics history
            with self.metrics_lock:
                for metric in self.metrics_history:
                    self.metrics_history[metric].clear()
            
            # Clear alerts
            with self.alerts_lock:
                self.alerts.clear()
                self.active_alerts.clear()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise ResourceCleanupError(f"Failed to clean up monitoring resources: {str(e)}") 