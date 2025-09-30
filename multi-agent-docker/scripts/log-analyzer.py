#!/usr/bin/env python3
"""
Log Analyzer for MCP Services
Analyzes structured logs and extracts patterns
"""

import json
import sys
import os
from datetime import datetime
from collections import defaultdict, Counter
import re

class LogAnalyzer:
    def __init__(self, log_dir='/app/mcp-logs'):
        self.log_dir = log_dir
        self.errors = []
        self.warnings = []
        self.performance_metrics = []
        self.security_events = []

    def parse_log_file(self, filename):
        """Parse a JSON log file"""
        filepath = os.path.join(self.log_dir, filename)
        if not os.path.exists(filepath):
            return []

        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def analyze_errors(self):
        """Extract and categorize errors"""
        error_entries = self.parse_log_file('error.json')

        error_patterns = defaultdict(int)
        for entry in error_entries:
            message = entry.get('message', '')
            # Extract error pattern
            pattern = re.sub(r'\d+', 'N', message)  # Replace numbers with N
            pattern = re.sub(r'[0-9a-f]{8,}', 'HEX', pattern)  # Replace hashes
            error_patterns[pattern] += 1

        return dict(error_patterns)

    def analyze_performance(self):
        """Extract performance metrics"""
        combined_entries = self.parse_log_file('combined.json')

        perf_metrics = []
        for entry in combined_entries:
            if 'operation' in entry and 'duration' in entry:
                perf_metrics.append({
                    'operation': entry['operation'],
                    'duration': entry['duration'],
                    'timestamp': entry.get('timestamp', '')
                })

        # Calculate statistics
        if perf_metrics:
            operations = defaultdict(list)
            for metric in perf_metrics:
                operations[metric['operation']].append(metric['duration'])

            stats = {}
            for op, durations in operations.items():
                stats[op] = {
                    'count': len(durations),
                    'avg': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
            return stats

        return {}

    def analyze_security(self):
        """Extract security events"""
        audit_entries = self.parse_log_file('security/audit.json')

        security_events = Counter()
        for entry in audit_entries:
            event = entry.get('event', 'unknown')
            security_events[event] += 1

        return dict(security_events)

    def detect_anomalies(self):
        """Detect anomalous patterns"""
        anomalies = []

        # Check for error spikes
        error_entries = self.parse_log_file('error.json')
        if len(error_entries) > 100:  # More than 100 errors
            anomalies.append({
                'type': 'error_spike',
                'count': len(error_entries),
                'severity': 'high'
            })

        # Check for slow operations
        perf_stats = self.analyze_performance()
        for op, stats in perf_stats.items():
            if stats['avg'] > 5000:  # Average > 5 seconds
                anomalies.append({
                    'type': 'slow_operation',
                    'operation': op,
                    'avg_duration': stats['avg'],
                    'severity': 'medium'
                })

        return anomalies

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("=" * 60)
        print("MCP Services Log Analysis Report")
        print(f"Generated: {datetime.now().isoformat()}")
        print("=" * 60)

        # Error Analysis
        print("\n[ERROR ANALYSIS]")
        error_patterns = self.analyze_errors()
        if error_patterns:
            print(f"Total unique error patterns: {len(error_patterns)}")
            print("\nTop 5 error patterns:")
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {count}x: {pattern[:80]}...")
        else:
            print("No errors found")

        # Performance Analysis
        print("\n[PERFORMANCE ANALYSIS]")
        perf_stats = self.analyze_performance()
        if perf_stats:
            print(f"Operations monitored: {len(perf_stats)}")
            for op, stats in perf_stats.items():
                print(f"  {op}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg: {stats['avg']:.2f}ms")
                print(f"    Min: {stats['min']:.2f}ms")
                print(f"    Max: {stats['max']:.2f}ms")
        else:
            print("No performance metrics found")

        # Security Analysis
        print("\n[SECURITY ANALYSIS]")
        security_events = self.analyze_security()
        if security_events:
            print(f"Security events detected: {sum(security_events.values())}")
            for event, count in sorted(security_events.items(), key=lambda x: x[1], reverse=True):
                print(f"  {event}: {count}")
        else:
            print("No security events found")

        # Anomaly Detection
        print("\n[ANOMALY DETECTION]")
        anomalies = self.detect_anomalies()
        if anomalies:
            print(f"Anomalies detected: {len(anomalies)}")
            for anomaly in anomalies:
                print(f"  [{anomaly['severity'].upper()}] {anomaly['type']}")
                for key, value in anomaly.items():
                    if key not in ['type', 'severity']:
                        print(f"    {key}: {value}")
        else:
            print("No anomalies detected")

        print("\n" + "=" * 60)

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/mcp-logs'
    analyzer = LogAnalyzer(log_dir)
    analyzer.generate_report()

if __name__ == '__main__':
    main()