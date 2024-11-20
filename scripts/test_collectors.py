import asyncio
from backend.collectors.windows_collector import WindowsEventCollector
from backend.collectors.network_collector import NetworkCollector

async def test_collectors():
    # Initialize collectors
    windows_collector = WindowsEventCollector({
        'log_types': ['System', 'Security', 'Application'],
        'collection_interval': 10
    })
    
    network_collector = NetworkCollector()
    
    # Test Windows event collection
    print("Testing Windows Event Collection...")
    async for event in windows_collector.collect_logs():
        print(f"Collected event: {event['message']}")
        break  # Just test one event
    
    # Test network collection
    print("\nTesting Network Collection...")
    async for event in network_collector.start_collection():
        print(f"Collected network event: {event}")
        break  # Just test one event

if __name__ == "__main__":
    asyncio.run(test_collectors())