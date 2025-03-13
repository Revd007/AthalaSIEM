using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository for user operations
    /// </summary>
    public class UserRepository : IUserRepository
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<UserRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="UserRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public UserRepository(ApplicationDbContext context, ILogger<UserRepository> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetByIdAsync(string id)
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .FirstOrDefaultAsync(u => u.Id == id);
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetByUsernameAsync(string username)
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .FirstOrDefaultAsync(u => u.Username == username);
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetByEmailAsync(string email)
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .FirstOrDefaultAsync(u => u.Email == email);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<UserModels>> GetAllAsync()
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<UserModels>> GetByRoleAsync(string roleName)
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .Where(u => u.UserRoles.Any(ur => ur.Role.Name == roleName))
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<UserModels> AddAsync(UserModels user)
        {
            _context.Users.Add(user);
            await _context.SaveChangesAsync();
            return user;
        }
        
        /// <inheritdoc/>
        public async Task<UserModels> UpdateAsync(UserModels user)
        {
            _context.Entry(user).State = EntityState.Modified;
            await _context.SaveChangesAsync();
            return user;
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteAsync(UserModels entity)
        {
            try
            {
                _context.Users.Remove(entity);
                await _context.SaveChangesAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting user");
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteByIdAsync(string id)
        {
            try
            {
                var user = await GetByIdAsync(id);
                if (user == null)
                    return false;

                return await DeleteAsync(user);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting user by ID: {Id}", id);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task DeleteAsync(string id)
        {
            await DeleteByIdAsync(id);
        }
        
        /// <inheritdoc/>
        public async Task<bool> IsInRoleAsync(string userId, string roleName)
        {
            var user = await GetByIdAsync(userId);
            return user?.UserRoles.Any(ur => ur.Role.Name == roleName) ?? false;
        }
        
        /// <inheritdoc/>
        public async Task AddRoleAsync(string userId, string roleName)
        {
            await AddToRoleAsync(userId, roleName);
        }
        
        /// <inheritdoc/>
        public async Task RemoveRoleAsync(string userId, string roleName)
        {
            await RemoveFromRoleAsync(userId, roleName);
        }

        public async Task<IEnumerable<string>> GetUserRolesAsync(string userId)
        {
            var user = await GetByIdAsync(userId);
            return user?.UserRoles.Select(ur => ur.Role.Name) ?? new List<string>();
        }

        public async Task<bool> AddToRoleAsync(string userId, string roleName)
        {
            try
            {
                var user = await GetByIdAsync(userId);
                if (user == null) return false;

                var role = await _context.Roles.FirstOrDefaultAsync(r => r.Name == roleName);
                if (role == null)
                {
                    role = new RoleModels
                    {
                        Name = roleName,
                        CreatedAt = DateTime.UtcNow
                    };
                    _context.Roles.Add(role);
                    await _context.SaveChangesAsync();
                }

                if (!await IsInRoleAsync(userId, roleName))
                {
                    user.UserRoles.Add(new UserRoleModels
                    {
                        UserId = userId,
                        RoleId = role.Id
                    });
                    await _context.SaveChangesAsync();
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding user {UserId} to role {RoleName}", userId, roleName);
                return false;
            }
        }

        public async Task<bool> RemoveFromRoleAsync(string userId, string roleName)
        {
            try
            {
                var user = await GetByIdAsync(userId);
                if (user == null) return false;

                var userRole = user.UserRoles.FirstOrDefault(ur => ur.Role.Name == roleName);
                if (userRole != null)
                {
                    user.UserRoles.Remove(userRole);
                    await _context.SaveChangesAsync();
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error removing user {UserId} from role {RoleName}", userId, roleName);
                return false;
            }
        }

        public async Task<IEnumerable<UserModels>> FindAsync(Expression<Func<UserModels, bool>> predicate)
        {
            return await _context.Users
                .Include(u => u.UserRoles)
                .ThenInclude(ur => ur.Role)
                .Where(predicate)
                .ToListAsync();
        }
    }
} 