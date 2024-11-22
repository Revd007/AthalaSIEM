import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import winreg
import win32serviceutil

class WindowsAgentInstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Athala SIEM - Windows Agent Installer")
        self.geometry("600x400")
        
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=20, pady=20, fill='both', expand=True)
        
        # Header
        ttk.Label(main_frame, text="Athala SIEM Agent", font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        # Configuration Frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Server Address
        ttk.Label(config_frame, text="SIEM Server Address:").pack(anchor='w', padx=5, pady=2)
        self.server_addr = ttk.Entry(config_frame)
        self.server_addr.pack(fill='x', padx=5, pady=2)
        self.server_addr.insert(0, "localhost:8080")
        
        # Log Types
        ttk.Label(config_frame, text="Log Types to Collect:").pack(anchor='w', padx=5, pady=2)
        self.log_types = ttk.Frame(config_frame)
        self.log_types.pack(fill='x', padx=5, pady=2)
        
        self.system_logs = tk.BooleanVar(value=True)
        self.security_logs = tk.BooleanVar(value=True)
        self.application_logs = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(self.log_types, text="System", variable=self.system_logs).pack(side='left', padx=5)
        ttk.Checkbutton(self.log_types, text="Security", variable=self.security_logs).pack(side='left', padx=5)
        ttk.Checkbutton(self.log_types, text="Application", variable=self.application_logs).pack(side='left', padx=5)
        
        # Install Button
        ttk.Button(main_frame, text="Install Agent", command=self.install_agent).pack(pady=20)
        
    def install_agent(self):
        try:
            # Create configuration
            config = {
                "server_address": self.server_addr.get(),
                "log_types": []
            }
            
            if self.system_logs.get(): config["log_types"].append("System")
            if self.security_logs.get(): config["log_types"].append("Security")
            if self.application_logs.get(): config["log_types"].append("Application")
            
            # Save configuration and install service
            self.save_config(config)
            self.install_windows_service()
            
            messagebox.showinfo("Success", "Agent installed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Installation failed: {str(e)}")
    
    def save_config(self, config):
        # Save configuration to registry
        try:
            key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\DonquixoteAthalaSIEM")
            winreg.SetValueEx(key, "ServerAddress", 0, winreg.REG_SZ, config["server_address"])
            winreg.SetValueEx(key, "LogTypes", 0, winreg.REG_MULTI_SZ, config["log_types"])
        finally:
            winreg.CloseKey(key)