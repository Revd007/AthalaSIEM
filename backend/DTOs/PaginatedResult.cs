using System.Collections.Generic;

namespace Backend.DTOs
{
    /// <summary>
    /// Represents a paginated result for API responses
    /// </summary>
    /// <typeparam name="T">Type of items in the result</typeparam>
    public class PaginatedResult<T>
    {
        /// <summary>
        /// Gets or sets the items
        /// </summary>
        public IEnumerable<T> Items { get; set; } = Array.Empty<T>();
        
        /// <summary>
        /// Gets or sets the total count of items
        /// </summary>
        public int TotalCount { get; set; }
        
        /// <summary>
        /// Gets or sets the page number
        /// </summary>
        public int Page { get; set; }
        
        /// <summary>
        /// Gets or sets the page size
        /// </summary>
        public int PageSize { get; set; }
        
        /// <summary>
        /// Gets or sets the total pages
        /// </summary>
        public int TotalPages { get; set; }
        
        /// <summary>
        /// Gets or sets whether there is a previous page
        /// </summary>
        public bool HasPreviousPage { get; set; }
        
        /// <summary>
        /// Gets or sets whether there is a next page
        /// </summary>
        public bool HasNextPage { get; set; }
    }
} 