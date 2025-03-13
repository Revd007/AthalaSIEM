using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Threading.Tasks;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Generic repository interface for data operations
    /// </summary>
    /// <typeparam name="TEntity">The entity type</typeparam>
    /// <typeparam name="TKey">The entity key type</typeparam>
    public interface IRepository<TEntity, TKey> where TEntity : class
    {
        /// <summary>
        /// Gets all entities
        /// </summary>
        /// <returns>All entities</returns>
        Task<IEnumerable<TEntity>> GetAllAsync();
        
        /// <summary>
        /// Gets an entity by its ID
        /// </summary>
        /// <param name="id">The entity ID</param>
        /// <returns>The entity if found, null otherwise</returns>
        Task<TEntity?> GetByIdAsync(TKey id);
        
        /// <summary>
        /// Finds entities based on a predicate
        /// </summary>
        /// <param name="predicate">The search predicate</param>
        /// <returns>Matching entities</returns>
        Task<IEnumerable<TEntity>> FindAsync(Expression<Func<TEntity, bool>> predicate);
        
        /// <summary>
        /// Adds a new entity
        /// </summary>
        /// <param name="entity">The entity to add</param>
        /// <returns>The added entity</returns>
        Task<TEntity> AddAsync(TEntity entity);
        
        /// <summary>
        /// Updates an existing entity
        /// </summary>
        /// <param name="entity">The entity to update</param>
        /// <returns>The updated entity</returns>
        Task<TEntity> UpdateAsync(TEntity entity);
        
        /// <summary>
        /// Deletes an entity
        /// </summary>
        /// <param name="entity">The entity to delete</param>
        /// <returns>True if deleted, false otherwise</returns>
        Task<bool> DeleteAsync(TEntity entity);
        
        /// <summary>
        /// Deletes an entity by its ID
        /// </summary>
        /// <param name="id">The entity ID</param>
        /// <returns>True if deleted, false otherwise</returns>
        Task<bool> DeleteByIdAsync(TKey id);
    }
} 