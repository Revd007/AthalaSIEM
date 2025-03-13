import os
from typing import Dict, List, Any, Optional
import re
import ipaddress
from datetime import datetime
import logging

class ThreatIndicators:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize indicator patterns
        self.indicators = {
            'network': self._load_network_indicators(),
            'file': self._load_file_indicators(),
            'behavior': self._load_behavior_indicators(),
            'registry': self._load_registry_indicators()
        }
    
    def _load_network_indicators(self) -> Dict[str, Any]:
        """Load network-based threat indicators"""
        return {
            'suspicious_ports': [4444, 666, 1337, 31337],  # Common malware ports
            'tor_exit_nodes': set(),  # To be updated from tor exit node list
            'known_c2_domains': set(),  # Command & Control domains
            'suspicious_protocols': ['irc', 'tftp'],
            'patterns': {
                'dns_tunneling': r'[a-zA-Z0-9]{25,}\.', # Long subdomain patterns
                'data_exfil': r'(\.zip|\.rar|\.7z|\.tar)$', # Compressed file extensions
                'shell_commands': r'(?:cmd|powershell|bash|sh)\s+', # Shell command patterns
            }
        }
    
    def _load_file_indicators(self) -> Dict[str, Any]:
        """Load file-based threat indicators"""
        return {
            'suspicious_extensions': ['.exe', '.dll', '.scr', '.bat', '.ps1'],
            'malware_strings': [
                'mimikatz', 'meterpreter', 'shellcode',
                'inject', 'payload', 'exploit'
            ],
            'suspicious_paths': [
                r'%temp%', r'%appdata%',
                r'windows\temp', r'system32\tasks'
            ],
            'file_signatures': {
                'webshell': [
                    rb'<?php.*eval\(',
                    rb'<%.*Response\.Write\(',
                    rb'<script.*runat=.*server'
                ]
            }
        }
    
    def _load_behavior_indicators(self) -> Dict[str, Any]:
        """Load behavior-based threat indicators"""
        return {
            'process_patterns': {
                'credential_access': [
                    'lsass.exe', 'mimikatz',
                    'wce.exe', 'pwdump'
                ],
                'persistence': [
                    'reg.exe add',
                    'schtasks /create',
                    'at \d{2}:'
                ],
                'evasion': [
                    'vssadmin delete',
                    'wevtutil cl',
                    'fsutil usn deletejournal'
                ]
            },
            'registry_patterns': {
                'autorun': [
                    r'SOFTWARE\Microsoft\Windows\CurrentVersion\Run',
                    r'SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce'
                ],
                'service_creation': [
                    r'SYSTEM\CurrentControlSet\Services'
                ]
            }
        }
    
    def _load_registry_indicators(self) -> Dict[str, Any]:
        """Load registry-based threat indicators"""
        return {
            'suspicious_keys': [
                r'SOFTWARE\Microsoft\Windows\CurrentVersion\Run',
                r'SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce',
                r'SYSTEM\CurrentControlSet\Services'
            ],
            'suspicious_values': [
                'script', 'http://', 'https://',
                '.exe', '.dll', '.ps1'
            ]
        }
    
    def check_indicators(self, 
                        data: Dict[str, Any],
                        indicator_type: str) -> List[Dict[str, Any]]:
        """Check data against specified indicator type"""
        matches = []
        
        if indicator_type not in self.indicators:
            self.logger.error(f"Unknown indicator type: {indicator_type}")
            return matches
            
        indicators = self.indicators[indicator_type]
        
        if indicator_type == 'network':
            matches.extend(self._check_network_indicators(data, indicators))
        elif indicator_type == 'file':
            matches.extend(self._check_file_indicators(data, indicators))
        elif indicator_type == 'behavior':
            matches.extend(self._check_behavior_indicators(data, indicators))
        elif indicator_type == 'registry':
            matches.extend(self._check_registry_indicators(data, indicators))
            
        return matches
    
    def _check_network_indicators(self, 
                                data: Dict[str, Any],
                                indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check network indicators"""
        matches = []
        
        # Check ports
        if 'port' in data and data['port'] in indicators['suspicious_ports']:
            matches.append({
                'type': 'suspicious_port',
                'value': data['port'],
                'severity': 'high'
            })
            
        # Check domains
        if 'domain' in data:
            for pattern in indicators['patterns']['dns_tunneling']:
                if re.search(pattern, data['domain']):
                    matches.append({
                        'type': 'dns_tunneling',
                        'value': data['domain'],
                        'severity': 'high'
                    })
                    
        return matches
    
    def _check_file_indicators(self,
                             data: Dict[str, Any],
                             indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check file indicators"""
        matches = []
        
        if 'path' in data:
            # Check extensions
            ext = os.path.splitext(data['path'])[1].lower()
            if ext in indicators['suspicious_extensions']:
                matches.append({
                    'type': 'suspicious_extension',
                    'value': ext,
                    'severity': 'medium'
                })
                
            # Check paths
            for sus_path in indicators['suspicious_paths']:
                if re.search(sus_path, data['path'], re.I):
                    matches.append({
                        'type': 'suspicious_path',
                        'value': data['path'],
                        'severity': 'medium'
                    })
                    
        return matches
    
    def update_indicators(self, 
                         indicator_type: str,
                         new_indicators: Dict[str, Any]):
        """Update threat indicators"""
        if indicator_type in self.indicators:
            self.indicators[indicator_type].update(new_indicators)
            self.logger.info(f"Updated {indicator_type} indicators")
        else:
            self.logger.error(f"Unknown indicator type: {indicator_type}")