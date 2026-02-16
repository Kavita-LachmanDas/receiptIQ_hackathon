"""
Email Alert Module
Sends automated overspending alert emails with beautiful HTML templates.
Uses Gmail SMTP with App Passwords for secure delivery.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime


class EmailAlerter:
    """Sends overspending alert emails with detailed spending reports."""

    def __init__(self):
        """Initialize with SMTP credentials from environment variables."""
        self.sender_email = os.getenv("SENDER_EMAIL", "")
        self.sender_password = os.getenv("SENDER_APP_PASSWORD", "")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))

    def is_configured(self) -> bool:
        """Check if email credentials are properly configured."""
        return bool(
            self.sender_email
            and self.sender_password
            and self.sender_email != "YOUR_EMAIL@gmail.com"
            and self.sender_password != "YOUR_APP_PASSWORD"
        )

    def should_send_alert(self, analysis_data: dict) -> bool:
        """Determine if overspending alerts exist that warrant an email."""
        overspending = analysis_data.get("overspending_alerts", [])
        return len(overspending) > 0

    def _build_html_email(self, recipient_name: str, categorization_data: dict,
                          analysis_data: dict, currency: str = "$") -> str:
        """Build a beautiful HTML email with spending alert details."""

        grand_total = categorization_data.get("grand_total", 0)
        total_items = categorization_data.get("total_items", 0)
        health_score = analysis_data.get("health_score", {})
        overspending = analysis_data.get("overspending_alerts", [])
        savings = analysis_data.get("savings_opportunities", [])
        top_items = analysis_data.get("top_expensive_items", [])
        category_pcts = categorization_data.get("category_percentages", {})

        score = health_score.get("score", 0)
        score_label = health_score.get("label", "N/A")
        score_color = health_score.get("color", "#6366F1")

        # Build overspending rows
        alert_rows = ""
        for alert in overspending:
            severity_color = "#EF4444" if alert["severity"] == "high" else "#F59E0B"
            severity_label = "HIGH" if alert["severity"] == "high" else "MEDIUM"
            alert_rows += f"""
            <tr>
                <td style="padding: 12px 16px; border-bottom: 1px solid #E2E8F0;">
                    <span style="background: {severity_color}; color: white; padding: 2px 8px; 
                           border-radius: 4px; font-size: 11px; font-weight: 700;">{severity_label}</span>
                </td>
                <td style="padding: 12px 16px; border-bottom: 1px solid #E2E8F0; font-weight: 600; color: #1E293B;">
                    {alert['category']}
                </td>
                <td style="padding: 12px 16px; border-bottom: 1px solid #E2E8F0; color: #64748B;">
                    {alert['actual_pct']}% of total
                </td>
                <td style="padding: 12px 16px; border-bottom: 1px solid #E2E8F0; color: {severity_color}; font-weight: 600;">
                    {alert['deviation_pct']}% over benchmark
                </td>
            </tr>
            """

        # Build category breakdown
        category_rows = ""
        for cat, data in category_pcts.items():
            bar_width = min(data["percentage"], 100)
            category_rows += f"""
            <tr>
                <td style="padding: 10px 16px; border-bottom: 1px solid #F1F5F9; font-weight: 500; color: #334155;">
                    {cat}
                </td>
                <td style="padding: 10px 16px; border-bottom: 1px solid #F1F5F9; color: #1E293B; font-weight: 600;">
                    {currency}{data['amount']:,.2f}
                </td>
                <td style="padding: 10px 16px; border-bottom: 1px solid #F1F5F9; width: 40%;">
                    <div style="background: #E2E8F0; border-radius: 10px; height: 8px; width: 100%;">
                        <div style="background: {data.get('color', '#6366F1')}; border-radius: 10px; 
                                height: 8px; width: {bar_width}%;"></div>
                    </div>
                    <span style="font-size: 12px; color: #64748B;">{data['percentage']}%</span>
                </td>
            </tr>
            """

        # Build top items list
        top_items_html = ""
        for i, item in enumerate(top_items[:5]):
            top_items_html += f"""
            <div style="display: flex; justify-content: space-between; padding: 8px 0; 
                        border-bottom: 1px solid #F1F5F9;">
                <span style="color: #334155;">{i+1}. {item['name']}</span>
                <span style="color: #1E293B; font-weight: 700; font-family: 'Courier New', monospace;">
                    {currency}{item['total']:,.2f}
                </span>
            </div>
            """

        # Build savings tips
        savings_html = ""
        for s in savings[:3]:
            savings_html += f"""
            <div style="background: #F0FDF4; border-left: 3px solid #22C55E; padding: 10px 16px; 
                        margin: 8px 0; border-radius: 0 8px 8px 0;">
                <span style="color: #166534;">💡 {s['tip']}</span>
            </div>
            """

        now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; background-color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            
            <!-- Container -->
            <div style="max-width: 640px; margin: 0 auto; padding: 20px;">
                
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
                            border-radius: 16px 16px 0 0; padding: 32px; text-align: center;">
                    <div style="font-size: 40px; margin-bottom: 8px;">🚨</div>
                    <h1 style="color: white; margin: 0; font-size: 24px; font-weight: 800;">
                        Overspending Alert
                    </h1>
                    <p style="color: #94A3B8; margin: 8px 0 0 0; font-size: 14px;">
                        ReceiptIQ has detected unusual spending patterns
                    </p>
                    <div style="display: inline-block; background: rgba(239,68,68,0.2); color: #FCA5A5; 
                                padding: 4px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; 
                                margin-top: 12px; border: 1px solid rgba(239,68,68,0.3);">
                        ⚠️ {len(overspending)} ALERT{'S' if len(overspending) > 1 else ''} DETECTED
                    </div>
                </div>

                <!-- Main Content -->
                <div style="background: white; padding: 32px; border: 1px solid #E2E8F0;">
                    
                    <p style="color: #475569; font-size: 15px; line-height: 1.6; margin-top: 0;">
                        Hi <strong>{recipient_name}</strong>,
                    </p>
                    <p style="color: #475569; font-size: 15px; line-height: 1.6;">
                        Our AI analysis of your latest receipt has identified 
                        <strong style="color: #EF4444;">{len(overspending)} overspending alert{'s' if len(overspending) > 1 else ''}</strong> 
                        that need your attention. Here's your detailed report:
                    </p>

                    <!-- Quick Stats -->
                    <div style="display: flex; gap: 12px; margin: 24px 0;">
                        <div style="flex: 1; background: #F8FAFC; border-radius: 12px; padding: 16px; text-align: center; 
                                    border: 1px solid #E2E8F0;">
                            <div style="font-size: 24px; font-weight: 800; color: #1E293B; font-family: 'Courier New', monospace;">
                                {currency}{grand_total:,.2f}
                            </div>
                            <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
                                Total Spent
                            </div>
                        </div>
                        <div style="flex: 1; background: #F8FAFC; border-radius: 12px; padding: 16px; text-align: center; 
                                    border: 1px solid #E2E8F0;">
                            <div style="font-size: 24px; font-weight: 800; color: #1E293B;">
                                {total_items}
                            </div>
                            <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
                                Items
                            </div>
                        </div>
                        <div style="flex: 1; background: #F8FAFC; border-radius: 12px; padding: 16px; text-align: center; 
                                    border: 1px solid {score_color}33;">
                            <div style="font-size: 24px; font-weight: 800; color: {score_color};">
                                {score}/100
                            </div>
                            <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
                                Health: {score_label}
                            </div>
                        </div>
                    </div>

                    <!-- Overspending Alerts Table -->
                    <h2 style="color: #1E293B; font-size: 18px; margin: 28px 0 12px 0; 
                               padding-bottom: 8px; border-bottom: 2px solid #EF4444;">
                        🚨 Overspending Alerts
                    </h2>
                    <table style="width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden; 
                                  border: 1px solid #E2E8F0;">
                        <thead>
                            <tr style="background: #F8FAFC;">
                                <th style="padding: 10px 16px; text-align: left; font-size: 11px; color: #64748B; 
                                           text-transform: uppercase; letter-spacing: 1px;">Severity</th>
                                <th style="padding: 10px 16px; text-align: left; font-size: 11px; color: #64748B; 
                                           text-transform: uppercase; letter-spacing: 1px;">Category</th>
                                <th style="padding: 10px 16px; text-align: left; font-size: 11px; color: #64748B; 
                                           text-transform: uppercase; letter-spacing: 1px;">% of Total</th>
                                <th style="padding: 10px 16px; text-align: left; font-size: 11px; color: #64748B; 
                                           text-transform: uppercase; letter-spacing: 1px;">Deviation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {alert_rows}
                        </tbody>
                    </table>

                    <!-- Category Breakdown -->
                    <h2 style="color: #1E293B; font-size: 18px; margin: 28px 0 12px 0; 
                               padding-bottom: 8px; border-bottom: 2px solid #6366F1;">
                        📊 Spending Breakdown
                    </h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tbody>
                            {category_rows}
                        </tbody>
                    </table>

                    <!-- Top Items -->
                    <h2 style="color: #1E293B; font-size: 18px; margin: 28px 0 12px 0; 
                               padding-bottom: 8px; border-bottom: 2px solid #F59E0B;">
                        💎 Most Expensive Items
                    </h2>
                    {top_items_html}

                    <!-- Savings Tips -->
                    <h2 style="color: #1E293B; font-size: 18px; margin: 28px 0 12px 0; 
                               padding-bottom: 8px; border-bottom: 2px solid #22C55E;">
                        💰 How to Save
                    </h2>
                    {savings_html if savings_html else '<p style="color: #64748B;">Keep up the smart spending!</p>'}

                    <!-- Action Required -->
                    <div style="background: linear-gradient(135deg, #FFF7ED, #FEF3C7); border: 1px solid #FDE68A; 
                                border-radius: 12px; padding: 20px; margin: 28px 0 0 0;">
                        <h3 style="color: #92400E; margin: 0 0 8px 0; font-size: 15px;">
                            🎯 Recommended Actions
                        </h3>
                        <ol style="color: #78350F; font-size: 14px; line-height: 1.8; margin: 0; padding-left: 20px;">
                            <li>Review the flagged categories and set spending limits</li>
                            <li>Look for store-brand alternatives for expensive items</li>
                            <li>Create a weekly budget plan to stay on track</li>
                        </ol>
                    </div>
                </div>

                <!-- Footer -->
                <div style="background: #1E293B; border-radius: 0 0 16px 16px; padding: 24px 32px; text-align: center;">
                    <div style="font-size: 20px; margin-bottom: 4px;">🧾</div>
                    <div style="color: #F1F5F9; font-weight: 700; font-size: 14px;">ReceiptIQ</div>
                    <div style="color: #64748B; font-size: 12px; margin-top: 4px;">
                        AI-Powered Receipt Analyzer
                    </div>
                    <div style="color: #475569; font-size: 11px; margin-top: 12px;">
                        Alert generated on {now}
                    </div>
                    <div style="color: #475569; font-size: 11px; margin-top: 4px;">
                        This is an automated alert from ReceiptIQ. Manage your alerts in the app settings.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def send_alert(self, recipient_email: str, recipient_name: str,
                   categorization_data: dict, analysis_data: dict,
                   currency: str = "$") -> dict:
        """
        Send an overspending alert email.

        Returns dict with 'success' (bool), 'message' (str).
        """
        if not self.is_configured():
            return {
                "success": False,
                "message": "Email not configured. Set SENDER_EMAIL and SENDER_APP_PASSWORD in .env file."
            }

        if not recipient_email or "@" not in recipient_email:
            return {
                "success": False,
                "message": "Invalid recipient email address."
            }

        try:
            # Build the email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 ReceiptIQ Overspending Alert — {currency}{categorization_data.get('grand_total', 0):,.2f} Spent"
            msg["From"] = f"ReceiptIQ Alerts <{self.sender_email}>"
            msg["To"] = recipient_email

            # Build HTML content
            html_content = self._build_html_email(
                recipient_name, categorization_data, analysis_data, currency
            )

            # Plain text fallback
            overspending = analysis_data.get("overspending_alerts", [])
            plain_text = f"ReceiptIQ Overspending Alert\n\n"
            plain_text += f"Hi {recipient_name},\n\n"
            plain_text += f"Total Spent: {currency}{categorization_data.get('grand_total', 0):,.2f}\n"
            plain_text += f"Health Score: {analysis_data.get('health_score', {}).get('score', 'N/A')}/100\n\n"
            plain_text += f"Overspending Alerts ({len(overspending)}):\n"
            for alert in overspending:
                plain_text += f"  - {alert['category']}: {alert['deviation_pct']}% over benchmark\n"
            plain_text += f"\nPlease review your spending habits in ReceiptIQ.\n"

            # Attach both versions
            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, msg.as_string())

            return {
                "success": True,
                "message": f"Alert email sent successfully to {recipient_email}!"
            }

        except smtplib.SMTPAuthenticationError:
            return {
                "success": False,
                "message": "SMTP authentication failed. Check your email and app password in .env file."
            }
        except smtplib.SMTPException as e:
            return {
                "success": False,
                "message": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }

