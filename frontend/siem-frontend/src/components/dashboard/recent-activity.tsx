import React, { useEffect } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { useInView } from 'react-intersection-observer'
import { Card, CardHeader, CardTitle } from '../ui/card'
import { Activity } from '../../types/activity'

async function fetchActivities(page: number): Promise<Activity[]> {
  const response = await fetch(`/api/activities?page=${page}`)
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
  } = useInfiniteQuery({
    queryKey: ['activities'],
    queryFn: ({ pageParam = 1 }) => fetchActivities(pageParam),
    getNextPageParam: (lastPage: Activity[], allPages: Activity[][]) => {
      return lastPage.length === 0 ? undefined : allPages.length + 1
    },
    initialPageParam: 1
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
                {/* Activity item content */}
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