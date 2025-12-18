#!/usr/bin/env python3
"""
Grafana Observability Stack MCP Server
Provides tools for Grafana, Prometheus, and Loki integration
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Configuration from environment
GRAFANA_URL = os.getenv('GRAFANA_URL', 'http://localhost:3000')
GRAFANA_API_KEY = os.getenv('GRAFANA_API_KEY', '')
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
LOKI_URL = os.getenv('LOKI_URL', 'http://localhost:3100')


class ObservabilityClient:
    """Client for observability stack APIs"""

    def __init__(self):
        self.grafana_url = GRAFANA_URL
        self.grafana_headers = {
            'Authorization': f'Bearer {GRAFANA_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.prom_url = PROMETHEUS_URL
        self.loki_url = LOKI_URL

    # Grafana Methods

    def grafana_request(self, path: str, method: str = 'GET', **kwargs) -> Dict:
        """Make Grafana API request"""
        url = urljoin(self.grafana_url, path)
        response = requests.request(
            method,
            url,
            headers=self.grafana_headers,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    def list_dashboards(self, folder: Optional[str] = None) -> List[Dict]:
        """List Grafana dashboards"""
        params = {'type': 'dash-db'}
        if folder:
            params['folderIds'] = folder

        results = self.grafana_request('/api/search', params=params)
        return results

    def get_dashboard(self, uid: str) -> Dict:
        """Get dashboard by UID"""
        return self.grafana_request(f'/api/dashboards/uid/{uid}')

    def search_dashboards(self, query: str) -> List[Dict]:
        """Search dashboards and folders"""
        params = {'query': query}
        return self.grafana_request('/api/search', params=params)

    def list_alerts(self, state: Optional[str] = None) -> List[Dict]:
        """List alert rules"""
        path = '/api/v1/provisioning/alert-rules'
        alerts = self.grafana_request(path)

        if state:
            # Filter by state if provided
            return [a for a in alerts if a.get('state') == state]
        return alerts

    def get_alert_status(self) -> Dict:
        """Get current alert states"""
        return self.grafana_request('/api/alerts')

    def list_datasources(self) -> List[Dict]:
        """List data sources"""
        return self.grafana_request('/api/datasources')

    # Prometheus Methods

    def prom_request(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Make Prometheus API request"""
        url = urljoin(self.prom_url, path)
        response = requests.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    def prom_query(self, query: str, time: Optional[str] = None) -> Dict:
        """Execute instant Prometheus query"""
        params = {'query': query}
        if time:
            params['time'] = time

        result = self.prom_request('/api/v1/query', params=params)
        return result['data']

    def prom_query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: str = '1m'
    ) -> Dict:
        """Execute range Prometheus query"""
        params = {
            'query': query,
            'start': start,
            'end': end,
            'step': step
        }

        result = self.prom_request('/api/v1/query_range', params=params)
        return result['data']

    def prom_series(self, match: str) -> List[Dict]:
        """List time series"""
        params = {'match[]': match}
        result = self.prom_request('/api/v1/series', params=params)
        return result['data']

    def prom_labels(self, label: Optional[str] = None) -> List[str]:
        """List labels or label values"""
        if label:
            path = f'/api/v1/label/{label}/values'
        else:
            path = '/api/v1/labels'

        result = self.prom_request(path)
        return result['data']

    def prom_targets(self) -> Dict:
        """Get scrape targets status"""
        result = self.prom_request('/api/v1/targets')
        return result['data']

    def prom_alerts(self) -> Dict:
        """Get Prometheus alerts"""
        result = self.prom_request('/api/v1/alerts')
        return result['data']

    # Loki Methods

    def loki_request(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Make Loki API request"""
        url = urljoin(self.loki_url, path)
        response = requests.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    def loki_query(
        self,
        query: str,
        limit: int = 100,
        since: str = '1h'
    ) -> Dict:
        """Execute LogQL query"""
        # Convert since to nanoseconds
        now = datetime.now()
        duration = self._parse_duration(since)
        start = int((now - duration).timestamp() * 1e9)
        end = int(now.timestamp() * 1e9)

        params = {
            'query': query,
            'limit': limit,
            'start': start,
            'end': end
        }

        result = self.loki_request('/loki/api/v1/query_range', params=params)
        return result['data']

    def loki_labels(self) -> List[str]:
        """List Loki labels"""
        result = self.loki_request('/loki/api/v1/labels')
        return result['data']

    def loki_series(self, match: str) -> List[Dict]:
        """List log streams"""
        params = {'match': match}
        result = self.loki_request('/loki/api/v1/series', params=params)
        return result['data']

    @staticmethod
    def _parse_duration(duration: str) -> timedelta:
        """Parse duration string to timedelta"""
        units = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days'
        }

        value = int(duration[:-1])
        unit = duration[-1]

        if unit not in units:
            raise ValueError(f"Invalid duration unit: {unit}")

        return timedelta(**{units[unit]: value})


# Initialize server and client
app = Server("grafana-monitor")
client = ObservabilityClient()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        # Grafana Tools
        Tool(
            name="grafana_dashboards",
            description="List Grafana dashboards, optionally filtered by folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Folder name to filter dashboards"
                    }
                }
            }
        ),
        Tool(
            name="grafana_dashboard_get",
            description="Get dashboard JSON definition by UID",
            inputSchema={
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "Dashboard UID"
                    }
                },
                "required": ["uid"]
            }
        ),
        Tool(
            name="grafana_alerts",
            description="List alert rules, optionally filtered by state",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Alert state: alerting, ok, pending, nodata, error"
                    }
                }
            }
        ),
        Tool(
            name="grafana_alert_status",
            description="Get current alert states",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="grafana_datasources",
            description="List all configured data sources",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="grafana_search",
            description="Search dashboards, folders, and panels",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),

        # Prometheus Tools
        Tool(
            name="prom_query",
            description="Execute instant PromQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PromQL query"
                    },
                    "time": {
                        "type": "string",
                        "description": "Query time (RFC3339 or Unix timestamp)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="prom_query_range",
            description="Execute range PromQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PromQL query"
                    },
                    "start": {
                        "type": "string",
                        "description": "Start time (RFC3339 or Unix timestamp)"
                    },
                    "end": {
                        "type": "string",
                        "description": "End time (RFC3339 or Unix timestamp)"
                    },
                    "step": {
                        "type": "string",
                        "description": "Query step duration (default: 1m)"
                    }
                },
                "required": ["query", "start", "end"]
            }
        ),
        Tool(
            name="prom_series",
            description="List time series matching selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "match": {
                        "type": "string",
                        "description": "Series selector"
                    }
                },
                "required": ["match"]
            }
        ),
        Tool(
            name="prom_labels",
            description="List label names or values for specific label",
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Label name to get values for"
                    }
                }
            }
        ),
        Tool(
            name="prom_targets",
            description="Get status of all Prometheus scrape targets",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="prom_alerts",
            description="Get active Prometheus alerts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # Loki Tools
        Tool(
            name="loki_query",
            description="Execute LogQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "LogQL query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of log lines (default: 100)"
                    },
                    "since": {
                        "type": "string",
                        "description": "Duration to look back (e.g., 1h, 30m, 1d)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="loki_labels",
            description="List all available log labels",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="loki_series",
            description="List log streams matching selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "match": {
                        "type": "string",
                        "description": "Stream selector"
                    }
                },
                "required": ["match"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls"""

    try:
        # Grafana tools
        if name == "grafana_dashboards":
            result = client.list_dashboards(folder=arguments.get('folder'))

        elif name == "grafana_dashboard_get":
            result = client.get_dashboard(uid=arguments['uid'])

        elif name == "grafana_alerts":
            result = client.list_alerts(state=arguments.get('state'))

        elif name == "grafana_alert_status":
            result = client.get_alert_status()

        elif name == "grafana_datasources":
            result = client.list_datasources()

        elif name == "grafana_search":
            result = client.search_dashboards(query=arguments['query'])

        # Prometheus tools
        elif name == "prom_query":
            result = client.prom_query(
                query=arguments['query'],
                time=arguments.get('time')
            )

        elif name == "prom_query_range":
            result = client.prom_query_range(
                query=arguments['query'],
                start=arguments['start'],
                end=arguments['end'],
                step=arguments.get('step', '1m')
            )

        elif name == "prom_series":
            result = client.prom_series(match=arguments['match'])

        elif name == "prom_labels":
            result = client.prom_labels(label=arguments.get('label'))

        elif name == "prom_targets":
            result = client.prom_targets()

        elif name == "prom_alerts":
            result = client.prom_alerts()

        # Loki tools
        elif name == "loki_query":
            result = client.loki_query(
                query=arguments['query'],
                limit=arguments.get('limit', 100),
                since=arguments.get('since', '1h')
            )

        elif name == "loki_labels":
            result = client.loki_labels()

        elif name == "loki_series":
            result = client.loki_series(match=arguments['match'])

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Run MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
