import pytest
import asyncio
import os

async def run_tests():
    # Test database
    pytest.main(['tests/database'])
    
    # Test AI Engine
    pytest.main(['tests/ai_engine'])
    
    # Test API
    pytest.main(['tests/api'])
    
    # Test Collectors
    pytest.main(['tests/collectors'])

if __name__ == "__main__":
    asyncio.run(run_tests())