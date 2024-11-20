import customtkinter as ctk
import sys
import os
import winreg
import subprocess

class InstallerGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("SIEM Solution Installer")
        self.root.geometry("600x400")
        
        # Installation path
        self.install_path = ctk.StringVar(value="C:\\Program Files\\SIEM-Solution")
        
        self.setup_ui()
    
    def setup_ui(self):
        # Installation path selector
        path_frame = ctk.CTkFrame(self.root)
        path_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(path_frame, text="Installation Path:").pack(side="left")
        ctk.CTkEntry(path_frame, textvariable=self.install_path, width=300).pack(side="left", padx=10)
        
        # Install button
        ctk.CTkButton(self.root, text="Install", command=self.install).pack(pady=20)
    
    def install(self):
        # Create installation directory
        os.makedirs(self.install_path.get(), exist_ok=True)
        
        # Copy files
        self.copy_files()
        
        # Set up auto-start
        self.setup_autostart()
        
        # Start service
        self.start_service()
        
        self.root.quit()
    
    def setup_autostart(self):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 
                             r"Software\Microsoft\Windows\CurrentVersion\Run")
        winreg.SetValueEx(key, "SIEM-Solution", 0, winreg.REG_SZ, 
                         os.path.join(self.install_path.get(), "siem_service.exe"))
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    installer = InstallerGUI()
    installer.run()