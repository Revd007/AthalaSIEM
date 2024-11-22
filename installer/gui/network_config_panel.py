from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from pathlib import Path
import ssl
import socket
import logging

class NetworkConfigPanel(QGroupBox):
    def __init__(self):
        super().__init__("Web Interface Configuration")
        self.logger = logging.getLogger(__name__)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout()
        
        # HTTP Configuration
        http_group = QGroupBox("HTTP Configuration")
        http_layout = QFormLayout()
        
        self.http_port = QSpinBox()
        self.http_port.setRange(1, 65535)
        self.http_port.setValue(8080)
        http_layout.addRow("HTTP Port:", self.http_port)
        
        http_group.setLayout(http_layout)
        
        # HTTPS Configuration
        https_group = QGroupBox("HTTPS Configuration")
        https_layout = QVBoxLayout()
        
        self.use_https = QCheckBox("Enable HTTPS")
        self.use_https.stateChanged.connect(self._toggle_https_options)
        
        https_form = QFormLayout()
        self.https_port = QSpinBox()
        self.https_port.setRange(1, 65535)
        self.https_port.setValue(8443)
        self.https_port.setEnabled(False)
        
        # SSL Certificate options
        self.cert_options = QComboBox()
        self.cert_options.addItems([
            "Generate self-signed certificate",
            "Use existing certificate"
        ])
        self.cert_options.setEnabled(False)
        self.cert_options.currentIndexChanged.connect(self._toggle_cert_fields)
        
        # Certificate file paths
        cert_path_layout = QGridLayout()
        self.cert_file = QLineEdit()
        self.key_file = QLineEdit()
        self.browse_cert = QPushButton("Browse")
        self.browse_key = QPushButton("Browse")
        
        self.cert_file.setEnabled(False)
        self.key_file.setEnabled(False)
        self.browse_cert.setEnabled(False)
        self.browse_key.setEnabled(False)
        
        self.browse_cert.clicked.connect(lambda: self._browse_file("certificate"))
        self.browse_key.clicked.connect(lambda: self._browse_file("key"))
        
        cert_path_layout.addWidget(QLabel("Certificate File:"), 0, 0)
        cert_path_layout.addWidget(self.cert_file, 0, 1)
        cert_path_layout.addWidget(self.browse_cert, 0, 2)
        cert_path_layout.addWidget(QLabel("Private Key File:"), 1, 0)
        cert_path_layout.addWidget(self.key_file, 1, 1)
        cert_path_layout.addWidget(self.browse_key, 1, 2)
        
        https_form.addRow("HTTPS Port:", self.https_port)
        https_form.addRow("SSL Certificate:", self.cert_options)
        
        https_layout.addWidget(self.use_https)
        https_layout.addLayout(https_form)
        https_layout.addLayout(cert_path_layout)
        
        https_group.setLayout(https_layout)
        
        # Add validation button
        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self._validate_config)
        
        layout.addWidget(http_group)
        layout.addWidget(https_group)
        layout.addWidget(self.validate_btn)
        
        self.setLayout(layout)
        
    def _toggle_https_options(self, state):
        """Enable/disable HTTPS options based on checkbox"""
        enabled = state == Qt.Checked
        self.https_port.setEnabled(enabled)
        self.cert_options.setEnabled(enabled)
        self._toggle_cert_fields(self.cert_options.currentIndex() if enabled else -1)
        
    def _toggle_cert_fields(self, index):
        """Toggle certificate file fields based on selection"""
        use_existing = index == 1 and self.use_https.isChecked()
        self.cert_file.setEnabled(use_existing)
        self.key_file.setEnabled(use_existing)
        self.browse_cert.setEnabled(use_existing)
        self.browse_key.setEnabled(use_existing)
        
    def _browse_file(self, file_type):
        """Browse for certificate or key file"""
        file_filter = "PEM files (*.pem);;All files (*.*)"
        title = f"Select SSL {'Certificate' if file_type == 'certificate' else 'Private Key'}"
        
        filename, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if filename:
            if file_type == 'certificate':
                self.cert_file.setText(filename)
            else:
                self.key_file.setText(filename)
                
    async def _validate_config(self):
        """Validate network configuration"""
        try:
            # Validate HTTP port
            if not self._is_port_available(self.http_port.value()):
                raise ValueError(f"HTTP port {self.http_port.value()} is already in use")
                
            # Validate HTTPS if enabled
            if self.use_https.isChecked():
                if not self._is_port_available(self.https_port.value()):
                    raise ValueError(f"HTTPS port {self.https_port.value()} is already in use")
                    
                if self.cert_options.currentIndex() == 1:
                    # Validate existing certificates
                    self._validate_certificates()
                    
            QMessageBox.information(self, "Validation", "Network configuration is valid!")
            
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            
    def _is_port_available(self, port: int) -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except:
            return False
            
    def _validate_certificates(self):
        """Validate SSL certificate files"""
        if self.cert_options.currentIndex() == 1:
            cert_path = self.cert_file.text()
            key_path = self.key_file.text()
            
            if not cert_path or not key_path:
                raise ValueError("Certificate and key files must be specified")
                
            try:
                ssl.create_default_context().load_cert_chain(cert_path, key_path)
            except Exception as e:
                raise ValueError(f"Invalid certificate or key file: {e}")
                
    def get_config(self) -> dict:
        """Get network configuration"""
        config = {
            'http_port': self.http_port.value(),
            'use_https': self.use_https.isChecked(),
            'https_port': self.https_port.value() if self.use_https.isChecked() else None,
            'ssl': {
                'enabled': self.use_https.isChecked(),
                'generate_cert': self.cert_options.currentIndex() == 0,
                'cert_file': self.cert_file.text() if self.cert_options.currentIndex() == 1 else None,
                'key_file': self.key_file.text() if self.cert_options.currentIndex() == 1 else None
            }
        }
        return config