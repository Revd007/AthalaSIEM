import React, { useEffect } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { useInView } from 'react-intersection-observer'
import { Card, CardHeader, CardTitle } from '../ui/card'
import { Activity } from '../../types/activity'
import { QueryFunctionContext, InfiniteData } from '@tanstack/react-query'

type ActivitiesResponse = Activity[]

// Define the query key type
type ActivityQueryKey = readonly ['activities']

async function fetchActivities(
  context: QueryFunctionContext<ActivityQueryKey, number>
): Promise<ActivitiesResponse> {
  const { pageParam } = context
  const response = await fetch(`/api/activities?page=${pageParam}`)
  if (!response.ok) throw new Error('Failed to fetch activities')
  return response.json()
}

export function RecentActivity() {
  const { ref, inView } = useInView()

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isLoading,
    isFetchingNextPage
  } = useInfiniteQuery<
    ActivitiesResponse,
    Error,
    InfiniteData<ActivitiesResponse>,
    ActivityQueryKey,
    number
  >({
    queryKey: ['activities'] as const,
    queryFn: fetchActivities,
    initialPageParam: 1,
    getNextPageParam: (lastPage, allPages) => {
      return lastPage.length === 0 ? undefined : allPages.length + 1
    }
  })

  useEffect(() => {
    if (inView && hasNextPage) {
      fetchNextPage()
    }
  }, [inView, fetchNextPage, hasNextPage])

  if (isLoading) return <div>Loading activities...</div>

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
      </CardHeader>
      <div className="p-6">
        {data?.pages.map((group, i) => (
          <React.Fragment key={i}>
            {group.map((activity) => (
              <div key={activity.id} className="py-2 border-b">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm font-medium">{activity.type}</span>
                    <span className="text-sm text-muted-foreground">
                      {activity.description}
                    </span>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {new Date(activity.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            ))}
          </React.Fragment>
        ))}
        <div ref={ref}>
          {isFetchingNextPage && <div>Loading more...</div>}
        </div>
      </div>
    </Card>
  )
}