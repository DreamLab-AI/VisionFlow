# WebSocket Protocol

## Connection

Connect to WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR_JWT_TOKEN');
```

## Message Format

All messages use JSON:

```json
{
  "type": "message_type",
  "data": {}
}
```

## Event Types

### Processing Status

```json
{
  "type": "processing.status",
  "data": {
    "jobId": "uuid",
    "status": "processing",
    "progress": 45
  }
}
```

### Notifications

```json
{
  "type": "notification",
  "data": {
    "id": "uuid",
    "title": "Processing Complete",
    "message": "Your job finished successfully"
  }
}
```

### Subscribe to Events

```json
{
  "type": "subscribe",
  "data": {
    "channels": ["projects.uuid", "notifications"]
  }
}
```

## Example

```javascript
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR_TOKEN');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    data: { channels: ['projects.123'] }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```
