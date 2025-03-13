using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Backend.Data;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Generic repository implementation for data operations
    /// </summary>
    /// <typeparam name="TEntity">The entity type</typeparam>
    /// <typeparam name="TKey">The entity key type</typeparam>
    public class Repository<TEntity, TKey> : IRepository<TEntity, TKey> where TEntity : class
    {
        /// <summary>
        /// The database context
        /// </summary>
        protected readonly ApplicationDbContext Context;
        
        /// <summary>
        /// The entity set
        /// </summary>
        protected readonly DbSet<TEntity> DbSet;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="Repository{TEntity, TKey}"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        public Repository(ApplicationDbContext context)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
            DbSet = context.Set<TEntity>();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<TEntity>> GetAllAsync()
        {
            return await DbSet.ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<TEntity?> GetByIdAsync(TKey id)
        {
            return await DbSet.FindAsync(id);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<TEntity>> FindAsync(Expression<Func<TEntity, bool>> predicate)
        {
            return await DbSet.Where(predicate).ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<TEntity> AddAsync(TEntity entity)
        {
            await DbSet.AddAsync(entity);
            await Context.SaveChangesAsync();
            return entity;
        }
        
        /// <inheritdoc/>
        public async Task<TEntity> UpdateAsync(TEntity entity)
        {
            Context.Entry(entity).State = EntityState.Modified;
            await Context.SaveChangesAsync();
            return entity;
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteAsync(TEntity entity)
        {
            DbSet.Remove(entity);
            return await Context.SaveChangesAsync() > 0;
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteByIdAsync(TKey id)
        {
            var entity = await GetByIdAsync(id);
            if (entity == null)
            {
                return false;
            }
            
            DbSet.Remove(entity);
            return await Context.SaveChangesAsync() > 0;
        }
        
        /// <inheritdoc/>
        public async Task<int> CountAsync()
        {
            return await DbSet.CountAsync();
        }
        
        /// <inheritdoc/>
        public async Task<int> CountAsync(Expression<Func<TEntity, bool>> predicate)
        {
            return await DbSet.CountAsync(predicate);
        }
        
        /// <inheritdoc/>
        public async Task<bool> AnyAsync()
        {
            return await DbSet.AnyAsync();
        }
        
        /// <inheritdoc/>
        public async Task<bool> AnyAsync(Expression<Func<TEntity, bool>> predicate)
        {
            return await DbSet.AnyAsync(predicate);
        }
    }
} 