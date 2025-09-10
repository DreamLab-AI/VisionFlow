#!/usr/bin/env python3
"""
Real-time Log Monitoring Dashboard for GPU Analytics System

This script provides a real-time monitoring dashboard that displays:
- Live GPU performance metrics
- Error rates and recovery attempts  
- Memory usage trends
- Kernel execution patterns
- System health indicators
"""

import json
import os
import sys
import time
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import curses
import argparse
from typing import Dict, List, Any, Optional, Deque
import logging
from dataclasses import dataclass
import psutil

@dataclass
class MetricSnapshot:
    timestamp: datetime
    gpu_kernels: Dict[str, float]  # kernel_name -> avg_time_us
    memory_usage_mb: float
    error_count: int
    recovery_attempts: int
    anomaly_count: int
    cpu_usage: float
    disk_usage: float

class LogTailer:
    """Tail log files for real-time monitoring"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.file_handle = None
        self.position = 0
        
    def start(self):
        """Start tailing the log file"""
        if self.log_file.exists():
            self.file_handle = open(self.log_file, 'r')
            # Start from end of file for real-time monitoring
            self.file_handle.seek(0, 2)
            self.position = self.file_handle.tell()
            
    def read_new_lines(self) -> List[str]:
        """Read new lines since last check"""
        if not self.file_handle or not self.log_file.exists():
            return []
            
        new_lines = []
        try:
            # Check if file was rotated
            current_size = self.log_file.stat().st_size
            if current_size < self.position:
                # File was rotated, reopen
                self.file_handle.close()
                self.start()
                return []
                
            self.file_handle.seek(self.position)
            new_lines = self.file_handle.readlines()
            self.position = self.file_handle.tell()
            
        except Exception as e:
            logging.error(f"Error reading {self.log_file}: {e}")
            # Try to recover
            self.start()
            
        return [line.strip() for line in new_lines if line.strip()]
        
    def close(self):
        """Close the file handle"""
        if self.file_handle:
            self.file_handle.close()

class RealtimeMonitor:
    """Real-time monitoring system"""
    
    def __init__(self, log_dir: str, update_interval: float = 1.0):
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        self.running = False
        
        # Metrics storage (keep last N snapshots)
        self.metrics_history: Deque[MetricSnapshot] = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.current_metrics = MetricSnapshot(
            timestamp=datetime.now(),
            gpu_kernels={},
            memory_usage_mb=0.0,
            error_count=0,
            recovery_attempts=0,
            anomaly_count=0,
            cpu_usage=0.0,
            disk_usage=0.0
        )
        
        # Log tailers
        self.tailers = {
            'gpu': LogTailer(self.log_dir / 'gpu.log'),
            'memory': LogTailer(self.log_dir / 'memory.log'),
            'performance': LogTailer(self.log_dir / 'performance.log'),
            'error': LogTailer(self.log_dir / 'error.log')
        }
        
        # Statistics
        self.kernel_timings = defaultdict(list)  # Keep last N timings per kernel
        self.error_rates = defaultdict(int)
        self.last_update = datetime.now()
        
        # Start tailers
        for tailer in self.tailers.values():
            tailer.start()
            
    def parse_log_entry(self, line: str) -> Optional[Dict]:
        """Parse JSON log entry"""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
            
    def update_metrics(self):
        """Update metrics from log files"""
        new_gpu_timings = {}
        memory_events = []
        new_errors = 0
        new_recoveries = 0
        new_anomalies = 0
        
        # Process GPU logs
        for line in self.tailers['gpu'].read_new_lines():
            entry = self.parse_log_entry(line)
            if entry and entry.get('gpu_metrics'):
                gpu_data = entry['gpu_metrics']
                
                if gpu_data.get('kernel_name') and gpu_data.get('execution_time_us'):
                    kernel_name = gpu_data['kernel_name']
                    exec_time = gpu_data['execution_time_us']
                    
                    # Update kernel timings (keep last 50 measurements)
                    self.kernel_timings[kernel_name].append(exec_time)
                    if len(self.kernel_timings[kernel_name]) > 50:
                        self.kernel_timings[kernel_name].pop(0)
                        
                    new_gpu_timings[kernel_name] = sum(self.kernel_timings[kernel_name]) / len(self.kernel_timings[kernel_name])
                    
                if gpu_data.get('performance_anomaly'):
                    new_anomalies += 1
                    
                if gpu_data.get('error_count'):
                    new_errors = max(new_errors, gpu_data['error_count'])
                    
                if gpu_data.get('recovery_attempts'):
                    new_recoveries = max(new_recoveries, gpu_data['recovery_attempts'])
                    
        # Process memory logs
        for line in self.tailers['memory'].read_new_lines():
            entry = self.parse_log_entry(line)
            if entry and entry.get('memory_usage_mb'):
                memory_events.append(entry['memory_usage_mb'])
                
        # Process error logs
        for line in self.tailers['error'].read_new_lines():
            entry = self.parse_log_entry(line)
            if entry and entry.get('level') == 'ERROR':
                self.error_rates[entry.get('component', 'unknown')] += 1
                
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=None)
        disk_usage = psutil.disk_usage(str(self.log_dir)).percent
        
        # Create new snapshot
        current_memory = memory_events[-1] if memory_events else self.current_metrics.memory_usage_mb
        
        self.current_metrics = MetricSnapshot(
            timestamp=datetime.now(),
            gpu_kernels=new_gpu_timings if new_gpu_timings else self.current_metrics.gpu_kernels,
            memory_usage_mb=current_memory,
            error_count=new_errors if new_errors > 0 else self.current_metrics.error_count,
            recovery_attempts=new_recoveries if new_recoveries > 0 else self.current_metrics.recovery_attempts,
            anomaly_count=self.current_metrics.anomaly_count + new_anomalies,
            cpu_usage=cpu_usage,
            disk_usage=disk_usage
        )
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
        self.last_update = datetime.now()
        
    def get_trend_data(self, metric: str, minutes: int = 5) -> List[float]:
        """Get trend data for a specific metric over the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if metric == 'memory':
            return [m.memory_usage_mb for m in recent_data]
        elif metric == 'errors':
            return [m.error_count for m in recent_data]
        elif metric == 'cpu':
            return [m.cpu_usage for m in recent_data]
        elif metric == 'anomalies':
            return [m.anomaly_count for m in recent_data]
        else:
            return []
            
    def get_kernel_performance_summary(self) -> Dict[str, Dict]:
        """Get kernel performance summary"""
        summary = {}
        
        for kernel_name, timings in self.kernel_timings.items():
            if timings:
                summary[kernel_name] = {
                    'avg_time_us': sum(timings) / len(timings),
                    'min_time_us': min(timings),
                    'max_time_us': max(timings),
                    'samples': len(timings),
                    'last_time_us': timings[-1] if timings else 0
                }
                
        return summary
        
    def start_monitoring(self):
        """Start the monitoring loop"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    self.update_metrics()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.update_interval)
                    
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        for tailer in self.tailers.values():
            tailer.close()

class CursesDashboard:
    """Curses-based real-time dashboard"""
    
    def __init__(self, monitor: RealtimeMonitor):
        self.monitor = monitor
        self.stdscr = None
        self.refresh_rate = 1.0  # seconds
        
    def create_bar_chart(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a simple ASCII bar chart"""
        if max_value == 0:
            filled = 0
        else:
            filled = int((value / max_value) * width)
            
        bar = '█' * filled + '░' * (width - filled)
        percentage = (value / max_value) * 100 if max_value > 0 else 0
        return f"{bar} {percentage:5.1f}%"
        
    def create_sparkline(self, data: List[float], width: int = 30) -> str:
        """Create a simple ASCII sparkline"""
        if not data or len(data) < 2:
            return '░' * width
            
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return '─' * width
            
        chars = '▁▂▃▄▅▆▇█'
        sparkline = ''
        
        # Sample data to fit width
        step = max(1, len(data) // width)
        sampled_data = data[::step][:width]
        
        for value in sampled_data:
            normalized = (value - min_val) / (max_val - min_val)
            char_index = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            sparkline += chars[char_index]
            
        return sparkline.ljust(width, '░')
        
    def draw_header(self, y: int):
        """Draw the dashboard header"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "GPU ANALYTICS MONITORING DASHBOARD"
        
        self.stdscr.addstr(y, 2, title, curses.A_BOLD | curses.A_REVERSE)
        self.stdscr.addstr(y, self.stdscr.getmaxyx()[1] - len(current_time) - 2, current_time)
        
    def draw_system_status(self, y: int):
        """Draw system status section"""
        self.stdscr.addstr(y, 2, "SYSTEM STATUS", curses.A_BOLD | curses.A_UNDERLINE)
        
        current = self.monitor.current_metrics
        uptime_str = f"Last Update: {current.timestamp.strftime('%H:%M:%S')}"
        
        self.stdscr.addstr(y + 1, 4, uptime_str)
        self.stdscr.addstr(y + 2, 4, f"CPU Usage: {self.create_bar_chart(current.cpu_usage, 100, 15)} {current.cpu_usage:5.1f}%")
        self.stdscr.addstr(y + 3, 4, f"Disk Usage: {self.create_bar_chart(current.disk_usage, 100, 15)} {current.disk_usage:5.1f}%")
        self.stdscr.addstr(y + 4, 4, f"Memory: {current.memory_usage_mb:8.1f} MB")
        
        return 6
        
    def draw_gpu_performance(self, y: int):
        """Draw GPU performance section"""
        self.stdscr.addstr(y, 2, "GPU KERNEL PERFORMANCE", curses.A_BOLD | curses.A_UNDERLINE)
        
        kernel_summary = self.monitor.get_kernel_performance_summary()
        
        if not kernel_summary:
            self.stdscr.addstr(y + 1, 4, "No GPU kernel data available")
            return 3
            
        row = y + 1
        self.stdscr.addstr(row, 4, f"{'Kernel Name':<20} {'Avg Time (μs)':<15} {'Samples':<10} {'Trend':<20}")
        row += 1
        
        for kernel_name, stats in list(kernel_summary.items())[:8]:  # Limit to 8 kernels
            if row >= self.stdscr.getmaxyx()[0] - 5:  # Leave space for other sections
                break
                
            # Get trend data for this kernel
            kernel_trend = []
            for snapshot in list(self.monitor.metrics_history)[-30:]:  # Last 30 data points
                if kernel_name in snapshot.gpu_kernels:
                    kernel_trend.append(snapshot.gpu_kernels[kernel_name])
                    
            sparkline = self.create_sparkline(kernel_trend, 20) if kernel_trend else '░' * 20
            
            self.stdscr.addstr(row, 4, f"{kernel_name:<20} {stats['avg_time_us']:<15.2f} {stats['samples']:<10} {sparkline}")
            row += 1
            
        return row - y + 1
        
    def draw_error_monitoring(self, y: int):
        """Draw error monitoring section"""
        self.stdscr.addstr(y, 2, "ERROR MONITORING", curses.A_BOLD | curses.A_UNDERLINE)
        
        current = self.monitor.current_metrics
        error_trend = self.monitor.get_trend_data('errors', 5)
        anomaly_trend = self.monitor.get_trend_data('anomalies', 5)
        
        self.stdscr.addstr(y + 1, 4, f"Total Errors: {current.error_count:8}")
        self.stdscr.addstr(y + 2, 4, f"Recovery Attempts: {current.recovery_attempts:5}")
        self.stdscr.addstr(y + 3, 4, f"Anomalies: {current.anomaly_count:11}")
        
        # Error rates by component
        row = y + 4
        if self.monitor.error_rates:
            self.stdscr.addstr(row, 4, "Error Rates by Component:")
            row += 1
            for component, count in list(self.monitor.error_rates.items())[:3]:
                self.stdscr.addstr(row, 6, f"{component}: {count}")
                row += 1
                
        return row - y + 1
        
    def draw_memory_trends(self, y: int):
        """Draw memory trends section"""
        self.stdscr.addstr(y, 2, "MEMORY TRENDS (Last 5 min)", curses.A_BOLD | curses.A_UNDERLINE)
        
        memory_trend = self.monitor.get_trend_data('memory', 5)
        
        if memory_trend:
            current_memory = memory_trend[-1] if memory_trend else 0
            max_memory = max(memory_trend) if memory_trend else 1
            min_memory = min(memory_trend) if memory_trend else 0
            
            sparkline = self.create_sparkline(memory_trend, 40)
            
            self.stdscr.addstr(y + 1, 4, f"Current: {current_memory:8.1f} MB")
            self.stdscr.addstr(y + 2, 4, f"Peak:    {max_memory:8.1f} MB") 
            self.stdscr.addstr(y + 3, 4, f"Min:     {min_memory:8.1f} MB")
            self.stdscr.addstr(y + 4, 4, f"Trend:   {sparkline}")
        else:
            self.stdscr.addstr(y + 1, 4, "No memory trend data available")
            
        return 6
        
    def draw_footer(self, y: int):
        """Draw dashboard footer"""
        footer_text = "Press 'q' to quit | 'r' to reset stats | Update interval: {:.1f}s".format(self.refresh_rate)
        max_y, max_x = self.stdscr.getmaxyx()
        
        if y < max_y - 1:
            self.stdscr.addstr(max_y - 1, 2, footer_text, curses.A_REVERSE)
        
    def run_dashboard(self, stdscr):
        """Main dashboard loop"""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(int(self.refresh_rate * 1000))  # Refresh timeout
        
        # Initialize colors if available
        if curses.has_colors():
            curses.start_colors()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            
        while True:
            try:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                
                if max_y < 20 or max_x < 80:
                    stdscr.addstr(0, 0, "Terminal too small! Need at least 80x20")
                    stdscr.refresh()
                    time.sleep(1)
                    continue
                
                current_y = 0
                
                # Draw sections
                self.draw_header(current_y)
                current_y += 2
                
                current_y += self.draw_system_status(current_y)
                current_y += 1
                
                current_y += self.draw_gpu_performance(current_y)
                current_y += 1
                
                current_y += self.draw_error_monitoring(current_y)
                current_y += 1
                
                current_y += self.draw_memory_trends(current_y)
                
                self.draw_footer(max_y)
                
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    # Reset statistics
                    self.monitor.error_rates.clear()
                    self.monitor.current_metrics.anomaly_count = 0
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                stdscr.addstr(0, 0, f"Error: {str(e)[:60]}")
                stdscr.refresh()
                time.sleep(1)
                
        curses.endwin()

def run_simple_dashboard(monitor: RealtimeMonitor):
    """Run a simple text-based dashboard for terminals without curses"""
    print("Starting simple dashboard... (Ctrl+C to exit)")
    print("=" * 80)
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            current = monitor.current_metrics
            kernel_summary = monitor.get_kernel_performance_summary()
            
            print("GPU ANALYTICS MONITORING DASHBOARD")
            print(f"Last Update: {current.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            print(f"\nSYSTEM STATUS:")
            print(f"  CPU Usage: {current.cpu_usage:5.1f}%")
            print(f"  Disk Usage: {current.disk_usage:5.1f}%")
            print(f"  Memory: {current.memory_usage_mb:8.1f} MB")
            
            print(f"\nGPU PERFORMANCE:")
            if kernel_summary:
                for kernel_name, stats in list(kernel_summary.items())[:5]:
                    print(f"  {kernel_name:<20}: {stats['avg_time_us']:>8.2f}μs (samples: {stats['samples']})")
            else:
                print("  No GPU kernel data available")
                
            print(f"\nERROR MONITORING:")
            print(f"  Total Errors: {current.error_count}")
            print(f"  Recovery Attempts: {current.recovery_attempts}")
            print(f"  Anomalies: {current.anomaly_count}")
            
            if monitor.error_rates:
                print(f"  Error Rates: {dict(list(monitor.error_rates.items())[:3])}")
                
            print("\n" + "=" * 80)
            print("Press Ctrl+C to exit")
            
            time.sleep(2)  # Update every 2 seconds for simple dashboard
            
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

def main():
    parser = argparse.ArgumentParser(description='GPU Analytics Real-time Log Monitor')
    parser.add_argument('--log-dir', default='./logs', help='Log directory path')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')
    parser.add_argument('--simple', action='store_true', help='Use simple text dashboard instead of curses')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.ERROR,  # Only show errors to avoid cluttering
        filename=os.path.join(args.log_dir, 'monitor.log'),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize monitor
    monitor = RealtimeMonitor(args.log_dir, args.interval)
    monitor.start_monitoring()
    
    try:
        if args.simple:
            run_simple_dashboard(monitor)
        else:
            dashboard = CursesDashboard(monitor)
            curses.wrapper(dashboard.run_dashboard)
            
    finally:
        monitor.stop_monitoring()
        print("Monitoring stopped.")

if __name__ == "__main__":
    main()