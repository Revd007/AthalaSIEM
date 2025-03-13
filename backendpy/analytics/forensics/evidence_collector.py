from typing import Dict, List, Any, Optional
import os
import json
import hashlib
import shutil
from datetime import datetime
import logging
from pathlib import Path

class EvidenceCollector:
    def __init__(self, case_dir: str = "evidence"):
        self.case_dir = Path(case_dir)
        self.logger = logging.getLogger(__name__)
        self.evidence_metadata = {}
        self._setup_evidence_directory()
        
    def _setup_evidence_directory(self):
        """Setup evidence directory structure"""
        try:
            # Create main evidence directory
            self.case_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.case_dir / "logs").mkdir(exist_ok=True)
            (self.case_dir / "files").mkdir(exist_ok=True)
            (self.case_dir / "memory").mkdir(exist_ok=True)
            (self.case_dir / "network").mkdir(exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to setup evidence directory: {e}")
    
    def collect_file_evidence(self, 
                            file_path: str, 
                            category: str,
                            description: str = "") -> Optional[Dict[str, Any]]:
        """Collect and hash file evidence"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None
                
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Copy file to evidence directory
            evidence_path = self.case_dir / "files" / f"{file_hash}_{os.path.basename(file_path)}"
            shutil.copy2(file_path, evidence_path)
            
            # Create metadata
            metadata = {
                'hash': file_hash,
                'original_path': file_path,
                'evidence_path': str(evidence_path),
                'category': category,
                'description': description,
                'collection_time': datetime.now().isoformat(),
                'file_size': os.path.getsize(file_path),
                'file_type': self._get_file_type(file_path)
            }
            
            # Store metadata
            self.evidence_metadata[file_hash] = metadata
            self._save_metadata()
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to collect file evidence: {e}")
            return None
    
    def collect_memory_dump(self, 
                          process_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Collect memory dump evidence"""
        try:
            from psutil import Process
            
            dump_path = self.case_dir / "memory" / f"memdump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
            
            if process_id:
                process = Process(process_id)
                # Collect specific process memory
                with open(dump_path, 'wb') as f:
                    f.write(process.memory_maps())
            else:
                # Use system tools for full memory dump
                if os.name == 'nt':  # Windows
                    os.system(f"winpmem -o {dump_path}")
                else:  # Linux
                    os.system(f"dd if=/dev/mem of={dump_path} bs=1MB")
            
            metadata = {
                'type': 'memory_dump',
                'process_id': process_id,
                'dump_path': str(dump_path),
                'collection_time': datetime.now().isoformat(),
                'size': os.path.getsize(dump_path)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to collect memory dump: {e}")
            return None
    
    def collect_network_capture(self, 
                              duration: int = 60,
                              interface: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Collect network capture evidence"""
        try:
            from scapy.all import sniff, wrpcap
            
            capture_path = self.case_dir / "network" / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
            
            # Capture network traffic
            packets = sniff(timeout=duration, iface=interface)
            wrpcap(str(capture_path), packets)
            
            metadata = {
                'type': 'network_capture',
                'capture_path': str(capture_path),
                'duration': duration,
                'interface': interface,
                'collection_time': datetime.now().isoformat(),
                'packet_count': len(packets)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to collect network capture: {e}")
            return None
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type"""
        import magic
        try:
            return magic.from_file(file_path, mime=True)
        except:
            return "unknown"
    
    def _save_metadata(self):
        """Save evidence metadata to file"""
        metadata_path = self.case_dir / "evidence_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.evidence_metadata, f, indent=2)