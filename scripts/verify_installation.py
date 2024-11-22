async def verify_installation():
    checks = [
        check_services(),
        check_database(),
        check_collectors(),
        check_ai_engine(),
        check_web_interface()
    ]
    
    return all(checks)