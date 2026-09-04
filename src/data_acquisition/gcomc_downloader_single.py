"""
Download H5 files from JAXA GPORTAL SFTP server.

Server: ftp.gportal.jaxa.jp
Port: 2051
Protocol: SFTP

Target directory: /standard/GCOM-C/GCOM-C.SGLI/L2.CRYOS.SIPR/3

Filename format: GC1SG1_YYYYMMDDmttt_gAAAA_LLx1x2_KKKKr_appp.h5
where AAAA = vvhh (vv: vertical tile 00-17, hh: horizontal tile 00-35)

Note:
This script was developed and tested on windows 11. Our server blocks port 2051, 
so I have not been able to verify if it works on linux systems.

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""

import paramiko
import os
import stat
import getpass
# from pathlib import Path
import time
import re
import socket

class SFTPDownloader:
    def __init__(self, host, port, username, password):
        """
        Initialize SFTP connection.
        
        Parameters:
        -----------
        host : str
            SFTP server hostname
        port : int
            SFTP server port
        username : str
            Username for authentication
        password : str
            Password for authentication
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ssh_client = None
        self.sftp_client = None
    
    def check_port_open(self, timeout=5):
        """
        Check if the SFTP port is open and reachable.
        
        Parameters:
        -----------
        timeout : int
            Connection timeout in seconds
        
        Returns:
        --------
        bool: True if port is open, False otherwise
        """
        print(f"\n{'='*60}")
        print("Port Connectivity Check")
        print(f"{'='*60}")
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"Timeout: {timeout}s")
        print("-" * 60)
        
        try:
            start_time = time.time()
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            elapsed = time.time() - start_time
            sock.close()
            
            print(f"✓ Port {self.port} is OPEN")
            print(f"✓ Connection established in {elapsed:.2f}s")
            print(f"{'='*60}\n")
            return True
            
        except socket.timeout:
            print(f"✗ Port {self.port} check TIMED OUT after {timeout}s")
            print("  This may indicate a firewall blocking the connection.")
            print(f"{'='*60}\n")
            return False
            
        except socket.gaierror as e:
            print(f"✗ DNS resolution FAILED: {e}")
            print(f"  Could not resolve hostname: {self.host}")
            print(f"{'='*60}\n")
            return False
            
        except ConnectionRefusedError:
            print(f"✗ Connection REFUSED")
            print(f"  Port {self.port} is closed or service is not running.")
            print(f"{'='*60}\n")
            return False
            
        except Exception as e:
            print(f"✗ Connection FAILED: {e}")
            print(f"{'='*60}\n")
            return False
        
    def connect(self):
        """Establish SFTP connection."""
        print(f"Connecting to {self.host}:{self.port}...")
        
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            self.ssh_client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=30
            )
            self.sftp_client = self.ssh_client.open_sftp()
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close SFTP connection."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        print("Disconnected.")
    
    def parse_gcomc_filename(self, filename):
        """
        Parse GCOM-C filename to extract tile numbers.
        
        Format: GC1SG1_YYYYMMDDmttt_gAAAA_LLx1x2_KKKKr_appp.h5
        where AAAA = vvhh (vv: vertical, hh: horizontal)
        
        Parameters:
        -----------
        filename : str
            GCOM-C H5 filename
        
        Returns:
        --------
        tuple: (vv, hh) as integers, or (None, None) if parsing fails
        """
        # Pattern: GC1SG1_YYYYMMDDmttt_g(vvhh)_...
        pattern = r'GC1SG1_\d{8}[A-Z]\d{2}[A-Z]_[A-Z](\d{2})(\d{2})_'
        match = re.match(pattern, filename)
        
        if match:
            vv = int(match.group(1))
            hh = int(match.group(2))
            return vv, hh
        
        return None, None
    
    def filter_by_tile(self, filename, vv_range=None, hh_range=None):
        """
        Check if file matches tile filter criteria.
        
        Parameters:
        -----------
        filename : str
            GCOM-C H5 filename
        vv_range : tuple or None
            (min_vv, max_vv) inclusive range for vertical tiles
        hh_range : tuple or None
            (min_hh, max_hh) inclusive range for horizontal tiles
        
        Returns:
        --------
        bool: True if file matches criteria
        """
        vv, hh = self.parse_gcomc_filename(filename)
        
        if vv is None or hh is None:
            return False
        
        # Check vv range
        if vv_range:
            if not (vv_range[0] <= vv <= vv_range[1]):
                return False
        
        # Check hh range
        if hh_range:
            if not (hh_range[0] <= hh <= hh_range[1]):
                return False
        
        return True
    
    def list_remote_files(self, remote_path, pattern=".h5", vv_range=None, hh_range=None):
        """
        Recursively list all files matching pattern and tile criteria in remote directory.
        
        Parameters:
        -----------
        remote_path : str
            Remote directory path
        pattern : str
            File extension or pattern to match
        vv_range : tuple or None
            (min_vv, max_vv) for vertical tile filter
        hh_range : tuple or None
            (min_hh, max_hh) for horizontal tile filter
        
        Returns:
        --------
        list of tuples: (remote_file_path, file_size, vv, hh)
        """
        files_found = []
        
        def recursive_list(path):
            try:
                items = self.sftp_client.listdir_attr(path)
                
                for item in items:
                    item_path = os.path.join(path, item.filename).replace('\\', '/')
                    
                    # Check if directory
                    if stat.S_ISDIR(item.st_mode):
                        print(f"Scanning directory: {item_path}")
                        recursive_list(item_path)
                    # Check if file matches pattern
                    elif item.filename.endswith(pattern):
                        # Apply tile filter
                        if self.filter_by_tile(item.filename, vv_range, hh_range):
                            vv, hh = self.parse_gcomc_filename(item.filename)
                            files_found.append((item_path, item.st_size, vv, hh))
                            print(f"  Found: {item.filename} (vv={vv:02d}, hh={hh:02d}, {item.st_size / 1024 / 1024:.2f} MB)")
                        
            except Exception as e:
                print(f"Error accessing {path}: {e}")
        
        filter_msg = []
        if vv_range:
            filter_msg.append(f"vv={vv_range[0]:02d}-{vv_range[1]:02d}")
        if hh_range:
            filter_msg.append(f"hh={hh_range[0]:02d}-{hh_range[1]:02d}")
        
        filter_str = f" ({', '.join(filter_msg)})" if filter_msg else ""
        print(f"Scanning {remote_path} for {pattern} files{filter_str}...")
        
        recursive_list(remote_path)
        
        return files_found
    
    def download_file(self, remote_path, local_path, resume=True):
        """
        Download a single file with progress indication.
        
        Parameters:
        -----------
        remote_path : str
            Remote file path
        local_path : str
            Local destination path
        resume : bool
            Resume partial downloads if True
        
        Returns:
        --------
        bool: True if successful, False otherwise
        """
        try:
            # Check if file already exists and is complete
            remote_size = self.sftp_client.stat(remote_path).st_size
            
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                if local_size == remote_size:
                    print(f"  Skipping (already complete): {os.path.basename(local_path)}")
                    return True
                elif resume and local_size < remote_size:
                    print(f"  Resuming download: {os.path.basename(local_path)} ({local_size}/{remote_size} bytes)")
                else:
                    print(f"  Re-downloading (size mismatch): {os.path.basename(local_path)}")
                    os.remove(local_path)
            
            # Create parent directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download with progress
            print(f"  Downloading: {os.path.basename(local_path)} ({remote_size / 1024 / 1024:.2f} MB)")
            
            start_time = time.time()
            downloaded = 0
            
            def progress_callback(transferred, total):
                nonlocal downloaded
                downloaded = transferred
                percent = (transferred / total) * 100 if total > 0 else 0
                mb_transferred = transferred / 1024 / 1024
                mb_total = total / 1024 / 1024
                print(f"\r    Progress: {percent:6.2f}% ({mb_transferred:.2f}/{mb_total:.2f} MB)", 
                      end='', flush=True)
            
            self.sftp_client.get(remote_path, local_path, callback=progress_callback)
            
            elapsed = time.time() - start_time
            speed = (remote_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            
            print(f"\n  Completed in {elapsed:.1f}s ({speed:.2f} MB/s)")
            return True
            
        except Exception as e:
            print(f"\n  Error downloading {remote_path}: {e}")
            return False
    
    def download_all_h5_files(self, remote_base_path, local_dest_path, 
                              preserve_structure=True, resume=True,
                              vv_range=None, hh_range=None):
        """
        Download all H5 files from remote directory tree with tile filtering.
        
        Parameters:
        -----------
        remote_base_path : str
            Remote base directory
        local_dest_path : str
            Local destination directory
        preserve_structure : bool
            If True, recreate remote directory structure locally
        resume : bool
            Resume partial downloads if True
        vv_range : tuple or None
            (min_vv, max_vv) for vertical tile filter
        hh_range : tuple or None
            (min_hh, max_hh) for horizontal tile filter
        
        Returns:
        --------
        dict: Statistics (total, successful, failed, skipped)
        """
        # Find all H5 files matching tile criteria
        files = self.list_remote_files(remote_base_path, pattern=".h5", 
                                       vv_range=vv_range, hh_range=hh_range)
        
        if not files:
            print("No H5 files found matching criteria.")
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
        print(f"\nFound {len(files)} H5 files matching criteria. Starting download...")
        
        stats = {'total': len(files), 'successful': 0, 'failed': 0, 'skipped': 0}
        
        for i, (remote_path, file_size, vv, hh) in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {remote_path} (vv={vv:02d}, hh={hh:02d})")
            
            # Determine local path
            if preserve_structure:
                # Preserve directory structure relative to base path
                # Remote paths use '/', convert to OS-appropriate separator for local paths
                rel_path = remote_path[len(remote_base_path):].lstrip('/')
                rel_path = rel_path.replace('/', os.sep)
                local_path = os.path.join(local_dest_path, rel_path)
            else:
                # Flatten structure
                filename = os.path.basename(remote_path.replace('/', os.sep))
                local_path = os.path.join(local_dest_path, filename)
            
            # Normalize path for the current OS
            local_path = os.path.normpath(local_path)
            
            # Check if already complete
            if os.path.exists(local_path) and os.path.getsize(local_path) == file_size:
                print(f"  Skipping (already complete)")
                stats['skipped'] += 1
                continue
            
            # Download
            if self.download_file(remote_path, local_path, resume=resume):
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        return stats


def main():
    """Interactive main function."""
    print("=" * 60)
    print("JAXA GPORTAL SFTP Downloader with Tile Filtering")
    print("=" * 60)
    
    # Server configuration
    host = "ftp.gportal.jaxa.jp"
    port = 2051
    
    # Get credentials
    username = input("Enter username: ")
    password = getpass.getpass("Enter password: ")
    
    # Get paths
    remote_path = input("Enter remote path [/standard/GCOM-C/GCOM-C.SGLI/L2.CRYOS.SIPR/3]: ").strip()
    if not remote_path:
        remote_path = "/standard/GCOM-C/GCOM-C.SGLI/L2.CRYOS.SIPR/3"
    
    local_path = input("Enter local destination path [/data_3/shunan_2/AU/hsa500m/GCOMC]: ").strip()
    if not local_path:
        local_path = "/data_3/shunan_2/AU/hsa500m/GCOMC"
    
    # Get tile filter ranges
    print("\nTile filtering (leave blank to skip filtering):")
    vv_min_str = input("  Enter minimum vertical tile (vv) [0]: ").strip()
    vv_max_str = input("  Enter maximum vertical tile (vv) [3]: ").strip()
    hh_min_str = input("  Enter minimum horizontal tile (hh) [15]: ").strip()
    hh_max_str = input("  Enter maximum horizontal tile (hh) [17]: ").strip()
    
    # Parse tile ranges
    vv_range = None
    hh_range = None
    
    if vv_min_str or vv_max_str:
        vv_min = int(vv_min_str) if vv_min_str else 0
        vv_max = int(vv_max_str) if vv_max_str else 3
        vv_range = (vv_min, vv_max)
        print(f"  Filtering: vv = {vv_min:02d} to {vv_max:02d}")
    
    if hh_min_str or hh_max_str:
        hh_min = int(hh_min_str) if hh_min_str else 15
        hh_max = int(hh_max_str) if hh_max_str else 17
        hh_range = (hh_min, hh_max)
        print(f"  Filtering: hh = {hh_min:02d} to {hh_max:02d}")
    
    preserve = input("\nPreserve directory structure? [y/n, default=y]: ").strip().lower()
    preserve_structure = preserve != 'n'
    
    # Initialize downloader
    downloader = SFTPDownloader(host, port, username, password)
    
    # Check if port is open before attempting connection
    if not downloader.check_port_open(timeout=10):
        print("⚠ Warning: Port appears to be unreachable.")
        proceed = input("Do you want to attempt connection anyway? [y/n]: ").strip().lower()
        if proceed != 'y':
            print("Aborted.")
            return
    
    if not downloader.connect():
        return
    
    try:
        # Download all H5 files with filtering
        stats = downloader.download_all_h5_files(
            remote_base_path=remote_path,
            local_dest_path=local_path,
            preserve_structure=preserve_structure,
            resume=True,
            vv_range=vv_range,
            hh_range=hh_range
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Download Summary:")
        print(f"  Total files: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Skipped (already complete): {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print("=" * 60)
        
    finally:
        downloader.disconnect()


if __name__ == "__main__":
    main()