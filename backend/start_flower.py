#!/usr/bin/env python3
"""
Start Flower monitoring for Celery.
"""

import os
import sys
import subprocess

def main():
    """Start Flower with proper configuration."""
    try:
        # Set environment variables
        os.environ.setdefault('FLOWER_PORT', '5555')
        os.environ.setdefault('FLOWER_BASIC_AUTH', 'admin:admin')
        
        # Start flower using celery command
        cmd = [
            'celery',
            '-A', 'celery_config',
            'flower',
            '--port=5555',
            '--basic_auth=admin:admin'
        ]
        
        print("Starting Flower monitoring...")
        print(f"Command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"Error starting Flower: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
