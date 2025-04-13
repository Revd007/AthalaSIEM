/**
 * Authentication Utilities
 * 
 * This file contains utility functions for handling authentication
 * and user session management.
 */

/**
 * Stores the user ID in the browser's storage
 * @param userId The user ID to store
 * @param rememberMe Whether to use localStorage (true) or sessionStorage (false)
 */
export function storeUserId(userId: string, rememberMe: boolean = false) {
  if (rememberMe) {
    localStorage.setItem('userId', userId);
  } else {
    sessionStorage.setItem('userId', userId);
  }
}

/**
 * Removes the user ID from browser storage
 */
export function clearUserId() {
  localStorage.removeItem('userId');
  sessionStorage.removeItem('userId');
}

/**
 * Gets the stored user ID
 * @returns The user ID or null if not found
 */
export function getStoredUserId(): string | null {
  return localStorage.getItem('userId') || sessionStorage.getItem('userId');
} 