// Tipe data untuk event dan alert
export interface Event {
    id: string;
    timestamp: string;
    source: string;
    event_type: string;
    severity: number;
    message: string;
}

export interface Alert {
    id: string;
    timestamp: string;
    title: string;
    severity: number;
    status: string;
}

export interface CollectorStatus {
    status: string;
    collected_events: Event[];
    count: number;
}

export interface CorrelationResult {
    status: string;
    test_event: Event;
    correlation_result: any;
}