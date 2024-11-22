from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
from ..core.installer_manager import InstallerManager
from ..core.progress_tracker import InstallationProgressTracker
from ..core.sql_detector import SQLServerDetector
from ..gui.company_info_panel import CompanyInfoPanel
from ..utils.email_notifier import InstallationNotifier
from datetime import datetime
from ..gui.network_config_panel import NetworkConfigPanel

class InstallerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AthalaSIEM Installer")
        self.setFixedSize(800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add logo and branding
        logo_label = QLabel()
        logo_pixmap = QPixmap("assets/logo.png")
        logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        
        # Add version and AI model info
        version_info = QLabel("Powered by Donquixote Athala AI")
        version_info.setStyleSheet("""
            QLabel {
                color: #336699;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        
        # Add to layout
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)
        layout.addWidget(version_info, alignment=Qt.AlignCenter)
        
        # Installation options
        self._create_install_options(layout)
        
        # Progress section
        self.progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to install...")
        self.progress_bar = QProgressBar()
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.detail_text)
        
        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)
        
        # Install button
        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self.start_installation)
        layout.addWidget(self.install_button)
        
        # Initialize progress tracker
        self.progress_tracker = InstallationProgressTracker()
        self.progress_tracker._callbacks.append(self.update_progress)
        
        # Database configuration group
        self.db_group = QGroupBox("Database Configuration")
        db_layout = QVBoxLayout()
        
        # SQL Server detection
        self.sql_detector = SQLServerDetector()
        sql_info = self.sql_detector.detect_sql_server()
        
        if sql_info['installed']:
            # Show existing instances
            self.instance_combo = QComboBox()
            for instance in sql_info['instances']:
                self.instance_combo.addItem(
                    f"{instance['instance_name']} ({instance['edition']})",
                    userData=instance
                )
            db_layout.addWidget(QLabel("Use existing SQL Server instance:"))
            db_layout.addWidget(self.instance_combo)
        else:
            # SQL Server installation options
            self.install_sql_radio = QRadioButton("Install SQL Server Express")
            self.skip_sql_radio = QRadioButton("Skip SQL Server installation (not recommended)")
            self.install_sql_radio.setChecked(True)
            
            db_layout.addWidget(self.install_sql_radio)
            db_layout.addWidget(self.skip_sql_radio)
            
        # Database credentials
        cred_group = QGroupBox("Database Credentials")
        cred_layout = QFormLayout()
        
        self.db_user = QLineEdit("revian_dbsiem")
        self.db_password = QLineEdit()
        self.db_password.setEchoMode(QLineEdit.Password)
        
        cred_layout.addRow("Database User:", self.db_user)
        cred_layout.addRow("Database Password:", self.db_password)
        
        cred_group.setLayout(cred_layout)
        db_layout.addWidget(cred_group)
        
        self.db_group.setLayout(db_layout)
        layout.addWidget(self.db_group)
        
        # Add company info panel
        self.company_info_panel = CompanyInfoPanel()
        layout.addWidget(self.company_info_panel)
        
        # Add checkbox for sending information
        self.send_info_checkbox = QCheckBox(
            "Send installation information to support team (Optional)"
        )
        layout.addWidget(self.send_info_checkbox)
        
        # Add network configuration panel
        self.network_config = NetworkConfigPanel()
        layout.addWidget(self.network_config)
        
    def update_progress(self, update: Dict[str, Any]) -> None:
        """Update progress display"""
        self.status_label.setText(update['step_name'])
        self.progress_bar.setValue(update['progress'])
        
        if update['message']:
            self.detail_text.append(
                f"[{update['timestamp']}] {update['message']}"
            )
        
    async def finish_installation(self):
        """Handle installation completion"""
        try:
            # ... other installation completion code ...
            
            # Send notification if checkbox is checked
            if self.send_info_checkbox.isChecked():
                company_info = {
                    'company_name': self.company_info_panel.company_name.text(),
                    'company_address': self.company_info_panel.company_address.toPlainText(),
                    'company_phone': self.company_info_panel.company_phone.text(),
                    'company_website': self.company_info_panel.company_website.text(),
                    'contact_name': self.company_info_panel.contact_name.text(),
                    'contact_email': self.company_info_panel.contact_email.text(),
                    'contact_phone': self.company_info_panel.contact_phone.text(),
                    'contact_position': self.company_info_panel.contact_position.text()
                }
                
                install_info = {
                    'install_date': datetime.now().isoformat(),
                    'version': self.config.get('version'),
                    'install_path': str(self.install_path),
                    'database_type': self.get_database_type()
                }
                
                notifier = InstallationNotifier()
                await notifier.send_installation_notification(company_info, install_info)
                
        except Exception as e:
            self.logger.error(f"Error in installation completion: {e}")
            QMessageBox.warning(
                self,
                "Warning",
                "Installation completed but failed to send information to support team."
            )
        
    async def install(self):
        try:
            # Get network configuration
            network_config = self.network_config.get_config()
            
            # Validate configuration before proceeding
            await self.network_config._validate_config()
            
            # Generate self-signed certificate if needed
            if (network_config['ssl']['enabled'] and 
                network_config['ssl']['generate_cert']):
                await self._generate_ssl_certificate()
                
            # Update configuration file
            self.config.update({
                'web_interface': network_config
            })
            
            # Continue with installation...
            
        except Exception as e:
            QMessageBox.critical(self, "Installation Error", str(e))