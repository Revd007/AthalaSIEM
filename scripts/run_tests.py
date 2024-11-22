# Buat file test_runner.py
import os
import pytest
import asyncio
from pathlib import Path

async def run_tests():
    # Test Backend
    backend_tests = [
        'backend/tests/ai_engine/',
        'backend/tests/analytics/',
        'backend/tests/collectors/',
        'backend/tests/api/'
    ]
    
    for test_path in backend_tests:
        print(f"\nRunning tests in {test_path}")
        pytest.main([test_path, '-v'])

    # Test Frontend
    os.chdir('frontend/siem-frontend')
    os.system('npm run test')

if __name__ == "__main__":
    asyncio.run(run_tests())