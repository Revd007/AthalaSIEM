from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from typing import Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import logging

class CompanyInfoPanel(QGroupBox):
    def __init__(self):
        super().__init__("Additional Information (Optional)")
        self.logger = logging.getLogger(__name__)
        self._init_ui()
        
    def _init_ui(self):
        layout = QFormLayout()
        
        # Company Information
        self.company_name = QLineEdit()
        self.company_address = QTextEdit()
        self.company_address.setMaximumHeight(60)
        self.company_phone = QLineEdit()
        self.company_website = QLineEdit()
        
        # Contact Person Information
        self.contact_name = QLineEdit()
        self.contact_email = QLineEdit()
        self.contact_phone = QLineEdit()
        self.contact_position = QLineEdit()
        
        # Add fields to layout
        layout.addRow("Company Name:", self.company_name)
        layout.addRow("Company Address:", self.company_address)
        layout.addRow("Company Phone:", self.company_phone)
        layout.addRow("Company Website:", self.company_website)
        layout.addRow(QLabel("")) # Spacer
        layout.addRow("Contact Person Name:", self.contact_name)
        layout.addRow("Contact Email:", self.contact_email)
        layout.addRow("Contact Phone:", self.contact_phone)
        layout.addRow("Position:", self.contact_position)
        
        self.setLayout(layout)