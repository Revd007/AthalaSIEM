import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
import logging
from config.security_config import EmailSettings

class EmailService:
    def __init__(self, settings: EmailSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
    async def send_email(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        html_content: Optional[str] = None
    ):
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.settings.SMTP_FROM_NAME} <{self.settings.SMTP_FROM_EMAIL}>"
            msg['To'] = ", ".join(recipients)
            
            # Add plain text
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML if provided
            if html_content:
                msg.attach(MIMEText(html_content, 'html'))
            
            # Connect to SMTP server
            with smtplib.SMTP(self.settings.SMTP_SERVER, self.settings.SMTP_PORT) as server:
                if self.settings.SMTP_USE_TLS:
                    server.starttls()
                server.login(self.settings.SMTP_USERNAME, self.settings.SMTP_PASSWORD)
                server.send_message(msg)
                
            self.logger.info(f"Email sent successfully to {recipients}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            raise
            
    async def send_security_alert(
        self,
        alert_type: str,
        details: dict,
        severity: int
    ):
        """Send security alert email"""
        if severity >= self.settings.ALERT_SEVERITY_THRESHOLD:
            subject = f"Security Alert: {alert_type} - Severity {severity}"
            body = self._format_alert_body(alert_type, details, severity)
            html = self._format_alert_html(alert_type, details, severity)
            
            await self.send_email(
                subject=subject,
                body=body,
                recipients=self.settings.ALERT_RECIPIENTS,
                html_content=html
            )
            
    def _format_alert_body(self, alert_type: str, details: dict, severity: int) -> str:
        return f"""
        Security Alert: {alert_type}
        Severity: {severity}
        Time: {datetime.utcnow()}
        
        Details:
        {details}
        """
        
    def _format_alert_html(self, alert_type: str, details: dict, severity: int) -> str:
        return f"""
        <html>
            <body>
                <h2>Security Alert: {alert_type}</h2>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Time:</strong> {datetime.utcnow()}</p>
                <h3>Details:</h3>
                <pre>{details}</pre>
            </body>
        </html>
        """ 