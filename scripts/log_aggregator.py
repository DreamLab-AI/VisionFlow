#!/usr/bin/env python3
"""
Log Aggregation and Analysis Script for CUDA GPU Analytics System

This script aggregates logs from different components, parses JSON-structured logs,
generates daily summaries, and provides performance analytics.
"""

import json
import os
import sys
import glob
import gzip
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

class LogAggregator:
    def __init__(self, log_dir: str, output_dir: str = None):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "aggregated"
        self.output_dir.mkdir(exist_ok=True)
        
        # Component log files
        self.log_files = {
            'gpu': self.log_dir / 'gpu.log',
            'server': self.log_dir / 'server.log',
            'client': self.log_dir / 'client.log',
            'analytics': self.log_dir / 'analytics.log',
            'memory': self.log_dir / 'memory.log',
            'network': self.log_dir / 'network.log',
            'performance': self.log_dir / 'performance.log',
            'error': self.log_dir / 'error.log'
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the aggregator itself"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'aggregator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON-structured log line"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            # Handle non-JSON lines (legacy format)
            if line.strip():
                return {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'component': 'unknown',
                    'message': line.strip(),
                    'metadata': None
                }
            return None
            
    def collect_logs(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """Collect logs from all components within date range"""
        logs = defaultdict(list)
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1)
        if end_date is None:
            end_date = datetime.now()
            
        self.logger.info(f"Collecting logs from {start_date} to {end_date}")
        
        for component, log_file in self.log_files.items():
            if log_file.exists():
                self.logger.info(f"Processing {component} logs from {log_file}")
                
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            entry = self.parse_log_line(line)
                            if entry:
                                # Parse timestamp and filter by date range
                                try:
                                    entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                                    # Make start_date and end_date timezone-aware for comparison
                                    if start_date.tzinfo is None:
                                        start_date = start_date.replace(tzinfo=entry_time.tzinfo)
                                    if end_date.tzinfo is None:
                                        end_date = end_date.replace(tzinfo=entry_time.tzinfo)
                                    
                                    if start_date <= entry_time <= end_date:
                                        logs[component].append(entry)
                                except (ValueError, KeyError):
                                    # Include entries without valid timestamps
                                    logs[component].append(entry)
                                    
                except Exception as e:
                    self.logger.error(f"Error reading {log_file}: {e}")
                    
                # Also check archived logs
                archived_dir = self.log_dir / "archived"
                if archived_dir.exists():
                    archived_files = glob.glob(str(archived_dir / f"{component}_*.log"))
                    for archived_file in archived_files:
                        try:
                            with open(archived_file, 'r') as f:
                                for line in f:
                                    entry = self.parse_log_line(line)
                                    if entry:
                                        try:
                                            entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                                            # Make start_date and end_date timezone-aware for comparison
                                            if start_date.tzinfo is None:
                                                start_date = start_date.replace(tzinfo=entry_time.tzinfo)
                                            if end_date.tzinfo is None:
                                                end_date = end_date.replace(tzinfo=entry_time.tzinfo)
                                            
                                            if start_date <= entry_time <= end_date:
                                                logs[component].append(entry)
                                        except (ValueError, KeyError):
                                            logs[component].append(entry)
                        except Exception as e:
                            self.logger.error(f"Error reading archived file {archived_file}: {e}")
            else:
                self.logger.warning(f"Log file not found: {log_file}")
                
        return dict(logs)
        
    def generate_gpu_performance_report(self, logs: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate GPU performance analytics report"""
        gpu_logs = logs.get('gpu', [])
        
        if not gpu_logs:
            return {"error": "No GPU logs found"}
            
        # Kernel performance analysis
        kernel_metrics = defaultdict(list)
        memory_events = []
        error_count = 0
        recovery_count = 0
        anomalies = []
        
        for entry in gpu_logs:
            if entry.get('gpu_metrics'):
                gpu_data = entry['gpu_metrics']
                
                # Collect kernel timing data
                if gpu_data.get('kernel_name') and gpu_data.get('execution_time_us'):
                    kernel_name = gpu_data['kernel_name']
                    exec_time = gpu_data['execution_time_us']
                    kernel_metrics[kernel_name].append({
                        'timestamp': entry['timestamp'],
                        'execution_time_us': exec_time,
                        'memory_mb': gpu_data.get('memory_allocated_mb', 0),
                        'peak_memory_mb': gpu_data.get('memory_peak_mb', 0),
                        'anomaly': gpu_data.get('performance_anomaly', False)
                    })
                    
                    if gpu_data.get('performance_anomaly'):
                        anomalies.append({
                            'kernel': kernel_name,
                            'timestamp': entry['timestamp'],
                            'execution_time_us': exec_time
                        })
                
                # Track errors and recovery
                if gpu_data.get('error_count'):
                    error_count = max(error_count, gpu_data['error_count'])
                if gpu_data.get('recovery_attempts'):
                    recovery_count = max(recovery_count, gpu_data['recovery_attempts'])
                    
            # Memory events
            if entry.get('component') == 'memory':
                memory_events.append({
                    'timestamp': entry['timestamp'],
                    'message': entry['message'],
                    'allocated_mb': entry.get('memory_usage_mb', 0)
                })
                
        # Calculate statistics for each kernel
        kernel_stats = {}
        for kernel, metrics in kernel_metrics.items():
            exec_times = [m['execution_time_us'] for m in metrics]
            memory_usage = [m['memory_mb'] for m in metrics]
            
            if exec_times:
                kernel_stats[kernel] = {
                    'count': len(exec_times),
                    'avg_time_us': np.mean(exec_times),
                    'min_time_us': np.min(exec_times),
                    'max_time_us': np.max(exec_times),
                    'std_time_us': np.std(exec_times),
                    'total_time_us': np.sum(exec_times),
                    'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                    'anomaly_rate': sum(1 for m in metrics if m.get('anomaly')) / len(metrics)
                }
                
        return {
            'summary': {
                'total_gpu_logs': len(gpu_logs),
                'unique_kernels': len(kernel_stats),
                'total_errors': error_count,
                'recovery_attempts': recovery_count,
                'performance_anomalies': len(anomalies)
            },
            'kernel_performance': kernel_stats,
            'anomalies': anomalies,
            'memory_events': memory_events[:50]  # Last 50 events
        }
        
    def generate_daily_summary(self, logs: Dict[str, List[Dict]], target_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive daily summary"""
        if target_date is None:
            target_date = datetime.now().date()
            
        summary = {
            'date': target_date.isoformat(),
            'generated_at': datetime.now().isoformat(),
            'components': {}
        }
        
        total_logs = 0
        error_summary = defaultdict(int)
        
        # Analyze each component
        for component, entries in logs.items():
            if not entries:
                continue
                
            component_summary = {
                'total_entries': len(entries),
                'levels': defaultdict(int),
                'errors': [],
                'warnings': [],
                'key_metrics': {}
            }
            
            for entry in entries:
                total_logs += 1
                level = entry.get('level', 'INFO')
                component_summary['levels'][level] += 1
                
                if level == 'ERROR':
                    error_summary[component] += 1
                    component_summary['errors'].append({
                        'timestamp': entry.get('timestamp'),
                        'message': entry.get('message', '')[:200]
                    })
                elif level == 'WARN':
                    component_summary['warnings'].append({
                        'timestamp': entry.get('timestamp'),
                        'message': entry.get('message', '')[:200]
                    })
                    
            # Component-specific metrics
            if component == 'gpu':
                gpu_report = self.generate_gpu_performance_report({component: entries})
                component_summary['key_metrics'] = gpu_report.get('summary', {})
                
            elif component == 'performance':
                perf_entries = [e for e in entries if e.get('execution_time_ms')]
                if perf_entries:
                    exec_times = [e['execution_time_ms'] for e in perf_entries]
                    component_summary['key_metrics'] = {
                        'operations_tracked': len(perf_entries),
                        'avg_execution_time_ms': np.mean(exec_times),
                        'slowest_operation_ms': np.max(exec_times),
                        'fastest_operation_ms': np.min(exec_times)
                    }
                    
            summary['components'][component] = component_summary
            
        summary['overall'] = {
            'total_log_entries': total_logs,
            'components_active': len([c for c in logs.keys() if logs[c]]),
            'error_distribution': dict(error_summary),
            'total_errors': sum(error_summary.values())
        }
        
        return summary
        
    def create_performance_charts(self, logs: Dict[str, List[Dict]]) -> List[str]:
        """Create performance visualization charts"""
        chart_files = []
        
        gpu_logs = logs.get('gpu', [])
        if not gpu_logs:
            return chart_files
            
        # Extract kernel performance data
        kernel_data = defaultdict(list)
        timestamps = []
        
        for entry in gpu_logs:
            if entry.get('gpu_metrics') and entry['gpu_metrics'].get('kernel_name'):
                gpu_data = entry['gpu_metrics']
                kernel_name = gpu_data['kernel_name']
                exec_time = gpu_data.get('execution_time_us', 0)
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                
                kernel_data[kernel_name].append((timestamp, exec_time))
                timestamps.append(timestamp)
                
        if not kernel_data:
            return chart_files
            
        # Kernel Performance Over Time Chart
        plt.figure(figsize=(12, 8))
        for kernel_name, data in kernel_data.items():
            times, exec_times = zip(*sorted(data))
            plt.plot(times, exec_times, label=kernel_name, marker='o', markersize=2)
            
        plt.xlabel('Time')
        plt.ylabel('Execution Time (μs)')
        plt.title('GPU Kernel Performance Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = self.output_dir / 'kernel_performance_timeline.png'
        plt.savefig(chart_path)
        plt.close()
        chart_files.append(str(chart_path))
        
        # Kernel Performance Distribution Chart
        if len(kernel_data) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (kernel_name, data) in enumerate(kernel_data.items()):
                if i >= 4:  # Limit to 4 kernels
                    break
                    
                exec_times = [d[1] for d in data]
                axes[i].hist(exec_times, bins=20, alpha=0.7)
                axes[i].set_title(f'{kernel_name} Performance Distribution')
                axes[i].set_xlabel('Execution Time (μs)')
                axes[i].set_ylabel('Frequency')
                
            plt.tight_layout()
            chart_path = self.output_dir / 'kernel_performance_distribution.png'
            plt.savefig(chart_path)
            plt.close()
            chart_files.append(str(chart_path))
            
        return chart_files
        
    def export_aggregated_logs(self, logs: Dict[str, List[Dict]], format='json'):
        """Export aggregated logs in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            output_file = self.output_dir / f'aggregated_logs_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
                
        elif format == 'csv':
            # Convert to flat structure for CSV
            all_entries = []
            for component, entries in logs.items():
                for entry in entries:
                    flat_entry = {
                        'component': component,
                        'timestamp': entry.get('timestamp', ''),
                        'level': entry.get('level', ''),
                        'message': entry.get('message', ''),
                        'execution_time_ms': entry.get('execution_time_ms', ''),
                        'memory_usage_mb': entry.get('memory_usage_mb', ''),
                    }
                    
                    # Add GPU metrics if present
                    if entry.get('gpu_metrics'):
                        gpu_data = entry['gpu_metrics']
                        flat_entry.update({
                            'gpu_kernel': gpu_data.get('kernel_name', ''),
                            'gpu_exec_time_us': gpu_data.get('execution_time_us', ''),
                            'gpu_memory_mb': gpu_data.get('memory_allocated_mb', ''),
                            'gpu_anomaly': gpu_data.get('performance_anomaly', ''),
                        })
                        
                    all_entries.append(flat_entry)
                    
            df = pd.DataFrame(all_entries)
            output_file = self.output_dir / f'aggregated_logs_{timestamp}.csv'
            df.to_csv(output_file, index=False)
            
        return output_file
        
    def run_aggregation(self, days: int = 1, generate_charts: bool = True, export_format: str = 'json'):
        """Run complete log aggregation process"""
        self.logger.info(f"Starting log aggregation for last {days} days")
        
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        # Collect logs
        logs = self.collect_logs(start_date, end_date)
        
        if not any(logs.values()):
            self.logger.warning("No logs found in the specified date range")
            return
            
        # Generate reports
        gpu_report = self.generate_gpu_performance_report(logs)
        daily_summary = self.generate_daily_summary(logs)
        
        # Save reports
        reports = {
            'gpu_performance': gpu_report,
            'daily_summary': daily_summary,
            'raw_logs': logs
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f'log_analysis_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
            
        self.logger.info(f"Analysis report saved to {report_file}")
        
        # Generate charts
        if generate_charts:
            chart_files = self.create_performance_charts(logs)
            self.logger.info(f"Generated {len(chart_files)} performance charts")
            
        # Export aggregated logs
        export_file = self.export_aggregated_logs(logs, export_format)
        self.logger.info(f"Exported aggregated logs to {export_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("LOG AGGREGATION SUMMARY")
        print("="*60)
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Total Components: {len(logs)}")
        print(f"Total Log Entries: {sum(len(entries) for entries in logs.values())}")
        
        if gpu_report.get('summary'):
            gpu_summary = gpu_report['summary']
            print(f"\nGPU Performance:")
            print(f"  - Unique Kernels: {gpu_summary.get('unique_kernels', 0)}")
            print(f"  - Total Errors: {gpu_summary.get('total_errors', 0)}")
            print(f"  - Recovery Attempts: {gpu_summary.get('recovery_attempts', 0)}")
            print(f"  - Performance Anomalies: {gpu_summary.get('performance_anomalies', 0)}")
            
        print(f"\nReports saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='GPU Analytics Log Aggregator')
    parser.add_argument('--log-dir', default='./logs', help='Log directory path')
    parser.add_argument('--output-dir', help='Output directory for aggregated data')
    parser.add_argument('--days', type=int, default=1, help='Number of days to aggregate')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    
    args = parser.parse_args()
    
    aggregator = LogAggregator(args.log_dir, args.output_dir)
    aggregator.run_aggregation(
        days=args.days,
        generate_charts=not args.no_charts,
        export_format=args.format
    )


if __name__ == "__main__":
    main()