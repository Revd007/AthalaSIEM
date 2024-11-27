export async function getAIStatus() {
    const response = await fetch('/api/ai/status');
    return response.json();
}

export async function analyzeEvent(eventData: any) {
    const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(eventData),
    });
    return response.json();
}