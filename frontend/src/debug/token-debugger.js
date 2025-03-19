// JWT Token Debugger - For development use only
// This script helps analyze JWT tokens stored in localStorage to diagnose authentication issues

// Function to parse a JWT token and return its payload
function parseJWT(token) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) {
      return { error: 'Invalid token format: Not a valid JWT token (needs 3 parts)' };
    }
    
    // Decode the payload (second part)
    const encodedPayload = parts[1];
    const decodedPayload = atob(encodedPayload);
    const payload = JSON.parse(decodedPayload);
    
    return payload;
  } catch (error) {
    return { error: `Failed to parse token: ${error.message}` };
  }
}

// Check if token exists in localStorage and parse it
function debugToken() {
  console.group('JWT Token Debug Info');
  
  const token = localStorage.getItem('token');
  if (!token) {
    console.warn('No token found in localStorage');
    console.groupEnd();
    return;
  }
  
  console.log('Token found:', token.substring(0, 15) + '...');
  
  // Parse the token
  const payload = parseJWT(token);
  
  if (payload.error) {
    console.error(payload.error);
    console.groupEnd();
    return;
  }
  
  // Show token expiration info
  const exp = payload.exp;
  const now = Math.floor(Date.now() / 1000);
  
  if (exp) {
    const expiryDate = new Date(exp * 1000);
    const isExpired = exp < now;
    
    console.log(`Token ${isExpired ? 'EXPIRED' : 'expires'} at: ${expiryDate.toLocaleString()}`);
    
    if (isExpired) {
      console.warn('⚠️ TOKEN IS EXPIRED - Authentication will fail');
    } else {
      const timeLeft = Math.floor((exp - now) / 60);
      console.log(`Time remaining: ${timeLeft} minutes`);
    }
  } else {
    console.warn('No expiration found in token');
  }
  
  // Show subject (user) info
  if (payload.sub) {
    console.log('Subject (user ID):', payload.sub);
  }
  
  // Look for role claims - this is critical for authorization
  const roleClaims = [];
  
  // Common claim names for roles
  const roleFields = [
    'role', 'roles', 'http://schemas.microsoft.com/ws/2008/06/identity/claims/role', 
    'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/role'
  ];
  
  for (const field of roleFields) {
    if (payload[field]) {
      if (Array.isArray(payload[field])) {
        roleClaims.push(...payload[field]);
      } else {
        roleClaims.push(payload[field]);
      }
    }
  }
  
  if (roleClaims.length > 0) {
    console.log('Roles found in token:', roleClaims);
    
    // Check if user has Admin or Operator role
    if (roleClaims.some(role => role === 'Admin' || role === 'Operator')) {
      console.log('✅ User has required role (Admin or Operator) for accessing agent endpoints');
    } else {
      console.warn('⚠️ User lacks required roles (Admin or Operator) for accessing agent endpoints');
    }
  } else {
    console.error('❌ No roles found in token - This will cause 403 Forbidden errors on endpoints requiring specific roles');
    console.log('Payload:', payload);
  }
  
  console.groupEnd();
  
  return {
    payload,
    hasRequiredRoles: roleClaims.some(role => role === 'Admin' || role === 'Operator'),
    isExpired: exp ? exp < now : false
  };
}

// Run the debugger automatically in development environments
if (process.env.NODE_ENV !== 'production') {
  console.log('Token Debugger initialized - run debugToken() in console to analyze the current JWT token');
  // Expose the function globally for console use
  window.debugToken = debugToken;
}

export { debugToken, parseJWT }; 