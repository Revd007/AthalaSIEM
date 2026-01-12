export interface AlertComment {
  id: string;
  comment: string;
  author: string;
  createdAt: string;
}

export interface Alert {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical' | string; // Backend uses string
  status: 'new' | 'in_progress' | 'resolved' | 'dismissed' | string; // Backend uses string
  source: string;
  timestamp: string; // Backend DateTime serialized as string
  agentId?: string;
  agentName?: string;
  ruleId?: string;
  ruleName?: string;
  assignedTo?: string;
  assignedToUserId?: string;
  assignedAt?: string;
  generatedBy?: string;
  lastUpdated?: string;
  lastUpdatedBy?: string;
  closedAt?: string;
  closedBy?: string;
  closeReason?: string;
  relatedLogIds?: string[];
  details?: Record<string, string>; // Backend uses Dictionary<string, string>
  comments?: AlertComment[];
  message?: string;
  resolutionNotes?: string;
  resolvedAt?: string;
  resolvedBy?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface AlertFilters {
  severity?: Alert['severity'];
  status?: Alert['status'];
  search?: string;
  agentId?: string;
  source?: string;
  ruleId?: string;
  assignedTo?: string;
  startTime?: string;
  endTime?: string;
  limit?: number;
  offset?: number;
  sortField?: string;
  sortDirection?: 'asc' | 'desc';
}

export interface PaginatedResult<T> {
  items: T[];
  totalCount: number;
  pageCount: number;
  currentPage: number;
  pageSize: number;
}

export interface AlertQueryParams {
  searchTerm?: string;
  severity?: string;
  status?: string;
  agentId?: string;
  source?: string;
  ruleId?: string;
  assignedTo?: string;
  startTime?: string;
  endTime?: string;
  limit?: number;
  offset?: number;
  sortField?: string;
  sortDirection?: string;
} 