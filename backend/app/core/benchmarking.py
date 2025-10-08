"""
Benchmarking and Performance Monitoring Framework

This module provides comprehensive benchmarking capabilities for the AI Assignment Checker system,
including timing decorators, metrics collection, performance monitoring, and detailed logging.
"""

import time
import functools
import threading
import psutil
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import logging
from pathlib import Path

# Configure benchmarking logger
benchmark_logger = logging.getLogger("benchmarking")
benchmark_logger.setLevel(logging.INFO)

# Create benchmark logs directory
benchmark_logs_dir = Path("logs/benchmarks")
benchmark_logs_dir.mkdir(parents=True, exist_ok=True)

# Add file handler for benchmark logs
benchmark_file_handler = logging.FileHandler(benchmark_logs_dir / "benchmark.log")
benchmark_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
benchmark_logger.addHandler(benchmark_file_handler)


@dataclass
class BenchmarkMetrics:
    """Data class for storing benchmark metrics."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    thread_id: Optional[int] = None
    session_id: Optional[str] = None
    file_size_bytes: Optional[int] = None
    function_count: Optional[int] = None
    qa_pair_count: Optional[int] = None
    embedding_count: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkCollector:
    """Collects and manages benchmark metrics across the application."""
    
    def __init__(self):
        self.metrics: List[BenchmarkMetrics] = []
        self.active_operations: Dict[str, BenchmarkMetrics] = {}
        self.lock = threading.Lock()
        self.session_metrics: Dict[str, List[BenchmarkMetrics]] = {}
        
    def start_operation(self, operation_name: str, session_id: str = None, **metadata) -> str:
        """Start timing an operation and return operation ID."""
        with self.lock:
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
            
            # Get current memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            metric = BenchmarkMetrics(
                operation_name=operation_name,
                start_time=datetime.now(),
                memory_start_mb=memory_mb,
                thread_id=threading.get_ident(),
                session_id=session_id,
                metadata=metadata
            )
            
            self.active_operations[operation_id] = metric
            
            benchmark_logger.info(f"[START] Started operation: {operation_name} (ID: {operation_id})")
            return operation_id
    
    def end_operation(self, operation_id: str, error_message: str = None, **final_metadata):
        """End timing an operation and collect final metrics."""
        with self.lock:
            if operation_id not in self.active_operations:
                benchmark_logger.warning(f"[WARNING] Operation ID not found: {operation_id}")
                return
            
            metric = self.active_operations[operation_id]
            metric.end_time = datetime.now()
            metric.duration_ms = (metric.end_time - metric.start_time).total_seconds() * 1000
            
            # Get final memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            metric.memory_end_mb = memory_info.rss / 1024 / 1024
            metric.memory_peak_mb = memory_info.rss / 1024 / 1024  # Will be updated with actual peak
            
            # Get CPU usage
            metric.cpu_percent = process.cpu_percent()
            
            # Update metadata
            metric.metadata.update(final_metadata)
            if error_message:
                metric.error_message = error_message
            
            # Move to completed metrics
            self.metrics.append(metric)
            del self.active_operations[operation_id]
            
            # Add to session metrics
            if metric.session_id:
                if metric.session_id not in self.session_metrics:
                    self.session_metrics[metric.session_id] = []
                self.session_metrics[metric.session_id].append(metric)
            
            # Log completion
            status = "[ERROR]" if error_message else "[SUCCESS]"
            benchmark_logger.info(
                f"{status} Completed operation: {metric.operation_name} "
                f"(ID: {operation_id}) - Duration: {metric.duration_ms:.2f}ms, "
                f"Memory: {metric.memory_end_mb:.1f}MB"
            )
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a specific session."""
        if session_id not in self.session_metrics:
            return {}
        
        metrics = self.session_metrics[session_id]
        if not metrics:
            return {}
        
        durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
        memory_usage = [m.memory_end_mb for m in metrics if m.memory_end_mb is not None]
        
        return {
            "session_id": session_id,
            "total_operations": len(metrics),
            "total_duration_ms": sum(durations),
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0,
            "average_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration_ms": m.duration_ms,
                    "memory_mb": m.memory_end_mb,
                    "error": m.error_message
                }
                for m in metrics
            ]
        }
    
    def export_benchmark_data(self, filename: str = None) -> str:
        """Export all benchmark data to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_data_{timestamp}.json"
        
        filepath = benchmark_logs_dir / filename
        
        # Convert metrics to serializable format
        serializable_metrics = []
        for metric in self.metrics:
            serializable_metrics.append({
                "operation_name": metric.operation_name,
                "start_time": metric.start_time.isoformat(),
                "end_time": metric.end_time.isoformat() if metric.end_time else None,
                "duration_ms": metric.duration_ms,
                "memory_start_mb": metric.memory_start_mb,
                "memory_end_mb": metric.memory_end_mb,
                "memory_peak_mb": metric.memory_peak_mb,
                "cpu_percent": metric.cpu_percent,
                "thread_id": metric.thread_id,
                "session_id": metric.session_id,
                "file_size_bytes": metric.file_size_bytes,
                "function_count": metric.function_count,
                "qa_pair_count": metric.qa_pair_count,
                "embedding_count": metric.embedding_count,
                "error_message": metric.error_message,
                "metadata": metric.metadata
            })
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_operations": len(self.metrics),
            "sessions": list(self.session_metrics.keys()),
            "metrics": serializable_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        benchmark_logger.info(f"📊 Exported benchmark data to: {filepath}")
        return str(filepath)


# Global benchmark collector instance
benchmark_collector = BenchmarkCollector()


def benchmark_operation(operation_name: str = None, session_id: str = None):
    """Decorator for benchmarking operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate operation name if not provided
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Extract session_id from kwargs if available
            session = session_id or kwargs.get('session_id')
            
            # Start benchmarking
            operation_id = benchmark_collector.start_operation(
                op_name, 
                session_id=session,
                function_name=func.__name__,
                module_name=func.__module__
            )
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract metadata from result if possible
                metadata = {}
                if hasattr(result, 'functions_evaluated'):
                    metadata['function_count'] = result.functions_evaluated
                if hasattr(result, 'matched_questions'):
                    metadata['qa_pair_count'] = result.matched_questions
                
                benchmark_collector.end_operation(operation_id, **metadata)
                return result
                
            except Exception as e:
                benchmark_collector.end_operation(
                    operation_id, 
                    error_message=str(e),
                    exception_type=type(e).__name__
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate operation name if not provided
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Extract session_id from kwargs if available
            session = session_id or kwargs.get('session_id')
            
            # Start benchmarking
            operation_id = benchmark_collector.start_operation(
                op_name, 
                session_id=session,
                function_name=func.__name__,
                module_name=func.__module__
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Extract metadata from result if possible
                metadata = {}
                if hasattr(result, 'functions_evaluated'):
                    metadata['function_count'] = result.functions_evaluated
                if hasattr(result, 'matched_questions'):
                    metadata['qa_pair_count'] = result.matched_questions
                
                benchmark_collector.end_operation(operation_id, **metadata)
                return result
                
            except Exception as e:
                benchmark_collector.end_operation(
                    operation_id, 
                    error_message=str(e),
                    exception_type=type(e).__name__
                )
                raise
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


@contextmanager
def benchmark_context(operation_name: str, session_id: str = None, **metadata):
    """Context manager for benchmarking code blocks."""
    operation_id = benchmark_collector.start_operation(
        operation_name, 
        session_id=session_id,
        **metadata
    )
    
    try:
        yield operation_id
        benchmark_collector.end_operation(operation_id)
    except Exception as e:
        benchmark_collector.end_operation(
            operation_id, 
            error_message=str(e),
            exception_type=type(e).__name__
        )
        raise


class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history: List[Dict[str, Any]] = []
        
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        benchmark_logger.info("📊 Started real-time performance monitoring")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        benchmark_logger.info("📊 Stopped real-time performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process(os.getpid())
        
        while self.monitoring:
            try:
                # Collect system metrics
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                metric = {
                    "timestamp": datetime.now().isoformat(),
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "cpu_percent": cpu_percent,
                    "thread_count": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "connections": len(process.connections())
                }
                
                self.metrics_history.append(metric)
                
                # Log significant changes
                if len(self.metrics_history) > 1:
                    prev_metric = self.metrics_history[-2]
                    memory_change = metric["memory_mb"] - prev_metric["memory_mb"]
                    
                    if abs(memory_change) > 50:  # 50MB change
                        benchmark_logger.info(
                            f"📈 Memory change: {memory_change:+.1f}MB "
                            f"(Current: {metric['memory_mb']:.1f}MB)"
                        )
                
                time.sleep(self.interval)
                
            except Exception as e:
                benchmark_logger.error(f"❌ Monitoring error: {e}")
                time.sleep(self.interval)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data."""
        if not self.metrics_history:
            return {}
        
        memory_values = [m["memory_mb"] for m in self.metrics_history]
        cpu_values = [m["cpu_percent"] for m in self.metrics_history]
        
        return {
            "monitoring_duration_seconds": len(self.metrics_history) * self.interval,
            "peak_memory_mb": max(memory_values),
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "peak_cpu_percent": max(cpu_values),
            "average_cpu_percent": sum(cpu_values) / len(cpu_values),
            "data_points": len(self.metrics_history)
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def generate_benchmark_report(session_id: str = None) -> str:
    """Generate a comprehensive benchmark report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"benchmark_report_{timestamp}.md"
    report_path = benchmark_logs_dir / report_filename
    
    with open(report_path, 'w') as f:
        f.write("# 🚀 AI Assignment Checker - Benchmark Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System overview
        f.write("## 📊 System Overview\n\n")
        f.write(f"- **Total Operations**: {len(benchmark_collector.metrics)}\n")
        f.write(f"- **Active Sessions**: {len(benchmark_collector.session_metrics)}\n")
        f.write(f"- **Active Operations**: {len(benchmark_collector.active_operations)}\n\n")
        
        # Session summaries
        if session_id:
            summary = benchmark_collector.get_session_summary(session_id)
            if summary:
                f.write(f"## 📈 Session: {session_id}\n\n")
                f.write(f"- **Total Operations**: {summary['total_operations']}\n")
                f.write(f"- **Total Duration**: {summary['total_duration_ms']:.2f}ms\n")
                f.write(f"- **Average Duration**: {summary['average_duration_ms']:.2f}ms\n")
                f.write(f"- **Peak Memory**: {summary['peak_memory_mb']:.1f}MB\n")
                f.write(f"- **Average Memory**: {summary['average_memory_mb']:.1f}MB\n\n")
        else:
            f.write("## 📈 Session Summaries\n\n")
            for session_id, summary in benchmark_collector.session_metrics.items():
                session_summary = benchmark_collector.get_session_summary(session_id)
                if session_summary:
                    f.write(f"### Session: {session_id}\n")
                    f.write(f"- **Operations**: {session_summary['total_operations']}\n")
                    f.write(f"- **Duration**: {session_summary['total_duration_ms']:.2f}ms\n")
                    f.write(f"- **Peak Memory**: {session_summary['peak_memory_mb']:.1f}MB\n\n")
        
        # Performance monitoring summary
        monitoring_summary = performance_monitor.get_monitoring_summary()
        if monitoring_summary:
            f.write("## 📊 Performance Monitoring\n\n")
            f.write(f"- **Monitoring Duration**: {monitoring_summary['monitoring_duration_seconds']:.1f}s\n")
            f.write(f"- **Peak Memory**: {monitoring_summary['peak_memory_mb']:.1f}MB\n")
            f.write(f"- **Average Memory**: {monitoring_summary['average_memory_mb']:.1f}MB\n")
            f.write(f"- **Peak CPU**: {monitoring_summary['peak_cpu_percent']:.1f}%\n")
            f.write(f"- **Average CPU**: {monitoring_summary['average_cpu_percent']:.1f}%\n\n")
        
        # Operation breakdown
        f.write("## 🔧 Operation Breakdown\n\n")
        operation_stats = {}
        for metric in benchmark_collector.metrics:
            if metric.operation_name not in operation_stats:
                operation_stats[metric.operation_name] = {
                    'count': 0,
                    'total_duration': 0,
                    'errors': 0,
                    'durations': []
                }
            
            stats = operation_stats[metric.operation_name]
            stats['count'] += 1
            if metric.duration_ms:
                stats['total_duration'] += metric.duration_ms
                stats['durations'].append(metric.duration_ms)
            if metric.error_message:
                stats['errors'] += 1
        
        for op_name, stats in operation_stats.items():
            f.write(f"### {op_name}\n")
            f.write(f"- **Count**: {stats['count']}\n")
            f.write(f"- **Total Duration**: {stats['total_duration']:.2f}ms\n")
            f.write(f"- **Average Duration**: {stats['total_duration']/stats['count']:.2f}ms\n")
            f.write(f"- **Errors**: {stats['errors']}\n")
            if stats['durations']:
                f.write(f"- **Min Duration**: {min(stats['durations']):.2f}ms\n")
                f.write(f"- **Max Duration**: {max(stats['durations']):.2f}ms\n")
            f.write("\n")
    
    benchmark_logger.info(f"📄 Generated benchmark report: {report_path}")
    return str(report_path)
