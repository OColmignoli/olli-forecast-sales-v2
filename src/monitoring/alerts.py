"""
Alert manager for the monitoring system.
"""
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
from .config import MonitoringConfig

class AlertManager:
    """Manager for handling monitoring alerts."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize alert manager.
        
        Args:
            env: Environment name
        """
        self.config = MonitoringConfig(env)
        self.logger = logging.getLogger(__name__)
        self._last_alert_time: Dict[str, datetime] = {}
    
    def _check_cooldown(self, alert_key: str) -> bool:
        """
        Check if alert is in cooldown period.
        
        Args:
            alert_key: Unique identifier for the alert
            
        Returns:
            True if alert can be sent, False if in cooldown
        """
        cooldown = self.config.alert_config['cooldown_period']
        last_time = self._last_alert_time.get(alert_key)
        
        if last_time is None:
            return True
            
        return datetime.now() - last_time > timedelta(seconds=cooldown)
    
    def _update_last_alert_time(self, alert_key: str) -> None:
        """
        Update last alert time.
        
        Args:
            alert_key: Unique identifier for the alert
        """
        self._last_alert_time[alert_key] = datetime.now()
    
    def _get_severity_level(self, metric_value: float) -> Optional[Dict[str, Any]]:
        """
        Get severity level based on metric value.
        
        Args:
            metric_value: Value to check
            
        Returns:
            Severity level configuration if matched, None otherwise
        """
        for level in sorted(
            self.config.alert_config['severity_levels'],
            key=lambda x: x['threshold'],
            reverse=True
        ):
            if metric_value >= level['threshold']:
                return level
        return None
    
    def _send_email_alert(
        self,
        subject: str,
        body: str,
        recipients: Optional[List[str]] = None
    ) -> None:
        """
        Send email alert.
        
        Args:
            subject: Email subject
            body: Email body
            recipients: Optional list of recipients
        """
        try:
            if not recipients:
                recipients = self.config.alert_config['email']['recipients']
            
            msg = MIMEText(body)
            msg['Subject'] = f"[Sales Forecast Alert] {subject}"
            msg['From'] = "alerts@salesforecast.com"
            msg['To'] = ", ".join(recipients)
            
            # In production, replace with actual SMTP configuration
            self.logger.info(f"Would send email: {msg.as_string()}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
            raise
    
    def _send_slack_alert(self, message: str) -> None:
        """
        Send Slack alert.
        
        Args:
            message: Alert message
        """
        try:
            slack_config = self.config.alert_config['slack']
            
            if not slack_config['enabled'] or not slack_config['webhook_url']:
                return
            
            payload = {
                'channel': slack_config['channel'],
                'text': message,
                'username': 'Sales Forecast Monitor'
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
            raise
    
    def send_metric_alert(
        self,
        metric_name: str,
        metric_value: float,
        threshold: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send alert for metric threshold violation.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value
            threshold: Threshold value
            details: Optional additional details
        """
        try:
            alert_key = f"{metric_name}_{threshold}"
            
            if not self._check_cooldown(alert_key):
                return
            
            severity = self._get_severity_level(metric_value / threshold)
            if not severity:
                return
            
            # Format alert message
            subject = (
                f"{severity['name'].upper()}: {metric_name} "
                f"threshold violation"
            )
            
            body = (
                f"Metric: {metric_name}\n"
                f"Current Value: {metric_value}\n"
                f"Threshold: {threshold}\n"
                f"Severity: {severity['name'].upper()}\n"
                f"Time: {datetime.now().isoformat()}\n"
            )
            
            if details:
                body += f"\nDetails:\n{json.dumps(details, indent=2)}"
            
            # Send alerts based on notification channels
            for channel in severity['notification_channels']:
                if channel == 'email' and self.config.alert_config['email']['enabled']:
                    self._send_email_alert(subject, body)
                elif channel == 'slack' and self.config.alert_config['slack']['enabled']:
                    self._send_slack_alert(f"*{subject}*\n```{body}```")
            
            self._update_last_alert_time(alert_key)
            self.logger.info(f"Sent {severity['name']} alert for {metric_name}")
            
        except Exception as e:
            self.logger.error(f"Error sending metric alert: {str(e)}")
            raise
    
    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        severity: str = 'critical',
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send alert for system error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            severity: Severity level
            details: Optional additional details
        """
        try:
            alert_key = f"error_{error_type}"
            
            if not self._check_cooldown(alert_key):
                return
            
            # Format alert message
            subject = f"{severity.upper()} Error: {error_type}"
            
            body = (
                f"Error Type: {error_type}\n"
                f"Message: {error_message}\n"
                f"Severity: {severity.upper()}\n"
                f"Time: {datetime.now().isoformat()}\n"
            )
            
            if details:
                body += f"\nDetails:\n{json.dumps(details, indent=2)}"
            
            # Send alerts based on severity
            if severity == 'critical':
                if self.config.alert_config['email']['enabled']:
                    self._send_email_alert(subject, body)
                if self.config.alert_config['slack']['enabled']:
                    self._send_slack_alert(f"*{subject}*\n```{body}```")
            elif severity == 'warning':
                if self.config.alert_config['email']['enabled']:
                    self._send_email_alert(subject, body)
            
            self._update_last_alert_time(alert_key)
            self.logger.info(f"Sent {severity} error alert for {error_type}")
            
        except Exception as e:
            self.logger.error(f"Error sending error alert: {str(e)}")
            raise
