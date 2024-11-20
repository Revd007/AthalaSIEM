from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os

class InstallerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AthalaSIEM Installer")
        self.setFixedSize(800, 500)
        self.setWindowIcon(QIcon("assets/icon.ico"))
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Banner
        banner = QLabel()
        banner.setPixmap(QPixmap("assets/banner.png"))
        layout.addWidget(banner)
        
        # Installation path
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(os.path.expandvars("%ProgramFiles%\\AthalaSIEM"))
        path_layout.addWidget(QLabel("Installation Path:"))
        path_layout.addWidget(self.path_input)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        
        # Port configuration
        port_layout = QHBoxLayout()
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(8080)
        port_layout.addWidget(QLabel("Web Interface Port:"))
        port_layout.addWidget(self.port_input)
        layout.addLayout(port_layout)
        
        # Install button
        install_btn = QPushButton("Install")
        install_btn.setFixedSize(200, 40)
        install_btn.clicked.connect(self.install)
        layout.addWidget(install_btn, alignment=Qt.AlignCenter)
        
        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Installation Directory")
        if path:
            self.path_input.setText(path)
            
    def install(self):
        # Installation logic here
        self.progress.setRange(0, 100)
        
        # Start installation in background thread
        self.thread = InstallationThread(
            self.path_input.text(),
            self.port_input.value()
        )
        self.thread.progress_updated.connect(self.progress.setValue)
        self.thread.finished.connect(self.installation_finished)
        self.thread.start()
        
    def installation_finished(self):
        QMessageBox.information(
            self,
            "Installation Complete",
            "AthalaSIEM has been installed successfully!\n"
            "The service will start automatically."
        )
        self.close()