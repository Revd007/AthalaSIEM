class InstallationErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors = []
        
    def handle_error(self, 
                    error: Exception,
                    step: str,
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle installation errors"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'error': str(error),
            'context': context,
            'recoverable': self._is_recoverable(error)
        }
        
        self.errors.append(error_info)
        self.logger.error(f"Installation error in {step}: {error}")
        
        # Try to recover if possible
        if error_info['recoverable']:
            recovery_result = self._attempt_recovery(error_info)
            error_info['recovery_result'] = recovery_result
            
        return error_info
        
    def _is_recoverable(self, error: Exception) -> bool:
        """Check if error is recoverable"""
        recoverable_errors = [
            'Login failed',
            'Network error',
            'Timeout',
            'Port in use'
        ]
        
        return any(err in str(error) for err in recoverable_errors)
        
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from error"""
        try:
            if 'Login failed' in error_info['error']:
                return self._handle_login_failure(error_info)
            elif 'Port in use' in error_info['error']:
                return self._handle_port_conflict(error_info)
            # Add more recovery handlers
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f"Recovery failed: {e}"
            }