import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface ChartData {
  timestamp: string;
  count: number;
  type: string;
}

interface EventsChartProps {
  data: ChartData[];
}

export function EventsChart({ data }: EventsChartProps) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis 
            dataKey="timestamp" 
            stroke="#94a3b8"
            fontSize={12}
            tickLine={false}
          />
          <YAxis 
            stroke="#94a3b8"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <CartesianGrid 
            strokeDasharray="3 3" 
            vertical={false}
            stroke="#e2e8f0"
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: 'white',
              border: 'none',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              padding: '12px'
            }}
          />
          <Area 
            type="monotone" 
            dataKey="count" 
            stroke="#8884d8" 
            fillOpacity={1} 
            fill="url(#colorEvents)" 
            name="Events"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}