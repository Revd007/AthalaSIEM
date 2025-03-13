using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace AthalaSIEM.Agent.Security
{
    /// <summary>
    /// Provides AES-256 encryption services with modern standards (GCM mode)
    /// </summary>
    public class AesEncryptionService : IEncryptionService
    {
        private readonly ILogger<AesEncryptionService> _logger;
        private const int KeySize = 32; // 256 bits
        private const int NonceSize = 12; // 96 bits for GCM
        private const int TagSize = 16; // 128 bits for GCM tag
        private const int SaltSize = 16; // 128 bits
        private const int Iterations = 10000; // PBKDF2 iterations
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AesEncryptionService"/> class
        /// </summary>
        /// <param name="logger">The logger</param>
        public AesEncryptionService(ILogger<AesEncryptionService> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Encrypts data using AES-256 GCM
        /// </summary>
        /// <param name="data">The data to encrypt</param>
        /// <param name="key">The encryption key</param>
        /// <returns>The encrypted data</returns>
        public byte[] Encrypt(byte[] data, byte[] key)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
                
            if (key == null || key.Length != KeySize)
                throw new ArgumentException($"Key must be {KeySize} bytes", nameof(key));
            
            try
            {
                // Generate a random nonce
                var nonce = new byte[NonceSize];
                using (var rng = RandomNumberGenerator.Create())
                {
                    rng.GetBytes(nonce);
                }
                
                // Encrypt the data with AES-GCM
                byte[] ciphertext;
                byte[] tag;
                
                using (var aesGcm = new AesGcm(key, TagSize))
                {
                    ciphertext = new byte[data.Length];
                    tag = new byte[TagSize];
                    
                    aesGcm.Encrypt(
                        nonce, 
                        data, 
                        ciphertext, 
                        tag, 
                        null // No associated data
                    );
                }
                
                // Combine nonce + tag + ciphertext for storage/transmission
                byte[] result = new byte[NonceSize + TagSize + ciphertext.Length];
                Buffer.BlockCopy(nonce, 0, result, 0, NonceSize);
                Buffer.BlockCopy(tag, 0, result, NonceSize, TagSize);
                Buffer.BlockCopy(ciphertext, 0, result, NonceSize + TagSize, ciphertext.Length);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error encrypting data");
                throw new CryptographicException("Encryption failed", ex);
            }
        }
        
        /// <summary>
        /// Decrypts data using AES-256 GCM
        /// </summary>
        /// <param name="encryptedData">The data to decrypt</param>
        /// <param name="key">The decryption key</param>
        /// <returns>The decrypted data</returns>
        public byte[] Decrypt(byte[] encryptedData, byte[] key)
        {
            if (encryptedData == null || encryptedData.Length <= NonceSize + TagSize)
                throw new ArgumentException("Encrypted data is invalid", nameof(encryptedData));
                
            if (key == null || key.Length != KeySize)
                throw new ArgumentException($"Key must be {KeySize} bytes", nameof(key));
            
            try
            {
                // Extract nonce, tag, and ciphertext
                byte[] nonce = new byte[NonceSize];
                byte[] tag = new byte[TagSize];
                byte[] ciphertext = new byte[encryptedData.Length - NonceSize - TagSize];
                
                Buffer.BlockCopy(encryptedData, 0, nonce, 0, NonceSize);
                Buffer.BlockCopy(encryptedData, NonceSize, tag, 0, TagSize);
                Buffer.BlockCopy(encryptedData, NonceSize + TagSize, ciphertext, 0, ciphertext.Length);
                
                // Decrypt the data with AES-GCM
                byte[] plaintext = new byte[ciphertext.Length];
                
                using (var aesGcm = new AesGcm(key, TagSize))
                {
                    aesGcm.Decrypt(
                        nonce, 
                        ciphertext, 
                        tag, 
                        plaintext, 
                        null // No associated data
                    );
                }
                
                return plaintext;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error decrypting data");
                throw new CryptographicException("Decryption failed", ex);
            }
        }
        
        /// <summary>
        /// Generates a random encryption key
        /// </summary>
        /// <returns>A random encryption key</returns>
        public byte[] GenerateKey()
        {
            var key = new byte[KeySize];
            using (var rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(key);
            }
            return key;
        }
        
        /// <summary>
        /// Creates a key from a password using PBKDF2
        /// </summary>
        /// <param name="password">The password</param>
        /// <param name="salt">The salt, or null to generate a new one</param>
        /// <returns>The derived key and salt</returns>
        public (byte[] Key, byte[] Salt) DeriveKeyFromPassword(string password, byte[]? salt = null)
        {
            if (string.IsNullOrEmpty(password))
                throw new ArgumentException("Password cannot be null or empty", nameof(password));
            
            // Generate salt if not provided
            if (salt == null)
            {
                salt = new byte[SaltSize];
                using (var rng = RandomNumberGenerator.Create())
                {
                    rng.GetBytes(salt);
                }
            }
            else if (salt.Length != SaltSize)
            {
                throw new ArgumentException($"Salt must be {SaltSize} bytes", nameof(salt));
            }
            
            // Derive key using PBKDF2
            byte[] key;
            using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, Iterations, HashAlgorithmName.SHA256))
            {
                key = pbkdf2.GetBytes(KeySize);
            }
            
            return (key, salt);
        }
        
        /// <summary>
        /// Computes a hash of the input data using SHA-256
        /// </summary>
        /// <param name="data">The data to hash</param>
        /// <returns>The hash value as a hexadecimal string</returns>
        public string ComputeHash(byte[] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
            
            using (var sha256 = SHA256.Create())
            {
                byte[] hash = sha256.ComputeHash(data);
                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
            }
        }
        
        /// <summary>
        /// Encrypts data with a password
        /// </summary>
        /// <param name="data">Data to encrypt</param>
        /// <param name="password">Password for encryption</param>
        /// <returns>Encrypted data with salt prepended</returns>
        public byte[] EncryptWithPassword(byte[] data, string password)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
                
            if (string.IsNullOrEmpty(password))
                throw new ArgumentException("Password cannot be null or empty", nameof(password));
            
            try
            {
                // Derive key from password
                var (key, salt) = DeriveKeyFromPassword(password);
                
                // Encrypt the data
                var encryptedData = Encrypt(data, key);
                
                // Combine salt + encrypted data
                var result = new byte[SaltSize + encryptedData.Length];
                Buffer.BlockCopy(salt, 0, result, 0, SaltSize);
                Buffer.BlockCopy(encryptedData, 0, result, SaltSize, encryptedData.Length);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error encrypting data with password");
                throw new CryptographicException("Password-based encryption failed", ex);
            }
        }
        
        /// <summary>
        /// Decrypts data with a password
        /// </summary>
        /// <param name="encryptedData">Encrypted data with salt prepended</param>
        /// <param name="password">Password for decryption</param>
        /// <returns>Decrypted data</returns>
        public byte[] DecryptWithPassword(byte[] encryptedData, string password)
        {
            if (encryptedData == null || encryptedData.Length <= SaltSize)
                throw new ArgumentException("Encrypted data is invalid", nameof(encryptedData));
                
            if (string.IsNullOrEmpty(password))
                throw new ArgumentException("Password cannot be null or empty", nameof(password));
            
            try
            {
                // Extract salt and encrypted data
                byte[] salt = new byte[SaltSize];
                byte[] encryptedDataWithoutSalt = new byte[encryptedData.Length - SaltSize];
                
                Buffer.BlockCopy(encryptedData, 0, salt, 0, SaltSize);
                Buffer.BlockCopy(encryptedData, SaltSize, encryptedDataWithoutSalt, 0, encryptedDataWithoutSalt.Length);
                
                // Derive key from password and salt
                var (key, _) = DeriveKeyFromPassword(password, salt);
                
                // Decrypt the data
                return Decrypt(encryptedDataWithoutSalt, key);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error decrypting data with password");
                throw new CryptographicException("Password-based decryption failed", ex);
            }
        }
        
        /// <summary>
        /// Computes a HMAC for data verification
        /// </summary>
        /// <param name="data">Data to sign</param>
        /// <param name="key">Key for HMAC</param>
        /// <returns>HMAC signature</returns>
        public byte[] ComputeHmac(byte[] data, byte[] key)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
                
            if (key == null || key.Length == 0)
                throw new ArgumentException("Key cannot be null or empty", nameof(key));
            
            using (var hmac = new HMACSHA256(key))
            {
                return hmac.ComputeHash(data);
            }
        }
        
        /// <summary>
        /// Verifies a HMAC signature
        /// </summary>
        /// <param name="data">Original data</param>
        /// <param name="key">Key used for HMAC</param>
        /// <param name="signature">HMAC signature to verify</param>
        /// <returns>True if signature is valid</returns>
        public bool VerifyHmac(byte[] data, byte[] key, byte[] signature)
        {
            var computedHmac = ComputeHmac(data, key);
            
            // Compare signatures in constant time to prevent timing attacks
            return CryptographicOperations.FixedTimeEquals(computedHmac, signature);
        }
    }
} 