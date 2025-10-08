from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from ....core.benchmarking import (
    benchmark_collector, 
    performance_monitor, 
    generate_benchmark_report,
    benchmark_logger
)
from ....core.logging import app_logger as logger

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])


@router.post("/start-monitoring")
async def start_performance_monitoring() -> Dict[str, Any]:
    """Start real-time performance monitoring."""
    try:
        performance_monitor.start_monitoring()
        logger.info("🚀 Started performance monitoring via API")
        
        return {
            "status": "success",
            "message": "Performance monitoring started",
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": performance_monitor.monitoring
        }
    except Exception as e:
        logger.error(f"❌ Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/stop-monitoring")
async def stop_performance_monitoring() -> Dict[str, Any]:
    """Stop real-time performance monitoring."""
    try:
        performance_monitor.stop_monitoring()
        logger.info("🛑 Stopped performance monitoring via API")
        
        return {
            "status": "success",
            "message": "Performance monitoring stopped",
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": performance_monitor.monitoring
        }
    except Exception as e:
        logger.error(f"❌ Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/status")
async def get_benchmark_status() -> Dict[str, Any]:
    """Get current benchmarking status."""
    try:
        monitoring_summary = performance_monitor.get_monitoring_summary()
        
        return {
            "status": "success",
            "monitoring_active": performance_monitor.monitoring,
            "total_operations": len(benchmark_collector.metrics),
            "active_operations": len(benchmark_collector.active_operations),
            "sessions": list(benchmark_collector.session_metrics.keys()),
            "monitoring_summary": monitoring_summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get benchmark status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/session/{session_id}")
async def get_session_metrics(session_id: str) -> Dict[str, Any]:
    """Get metrics for a specific session."""
    try:
        summary = benchmark_collector.get_session_summary(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "status": "success",
            "session_id": session_id,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session metrics: {str(e)}")


@router.get("/sessions")
async def get_all_sessions() -> Dict[str, Any]:
    """Get summary of all sessions."""
    try:
        session_summaries = {}
        for session_id in benchmark_collector.session_metrics.keys():
            session_summaries[session_id] = benchmark_collector.get_session_summary(session_id)
        
        return {
            "status": "success",
            "sessions": session_summaries,
            "total_sessions": len(session_summaries),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get all sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@router.post("/generate-report")
async def generate_report(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a comprehensive benchmark report."""
    try:
        # Generate report in background
        background_tasks.add_task(generate_benchmark_report, session_id)
        
        return {
            "status": "success",
            "message": "Report generation started",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.post("/export-data")
async def export_benchmark_data(filename: Optional[str] = None) -> Dict[str, Any]:
    """Export all benchmark data to JSON file."""
    try:
        filepath = benchmark_collector.export_benchmark_data(filename)
        
        return {
            "status": "success",
            "message": "Benchmark data exported",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to export data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/operations")
async def get_operation_breakdown() -> Dict[str, Any]:
    """Get breakdown of all operations."""
    try:
        operation_stats = {}
        for metric in benchmark_collector.metrics:
            if metric.operation_name not in operation_stats:
                operation_stats[metric.operation_name] = {
                    'count': 0,
                    'total_duration': 0,
                    'errors': 0,
                    'durations': [],
                    'memory_usage': []
                }
            
            stats = operation_stats[metric.operation_name]
            stats['count'] += 1
            if metric.duration_ms:
                stats['total_duration'] += metric.duration_ms
                stats['durations'].append(metric.duration_ms)
            if metric.memory_end_mb:
                stats['memory_usage'].append(metric.memory_end_mb)
            if metric.error_message:
                stats['errors'] += 1
        
        # Calculate averages
        for op_name, stats in operation_stats.items():
            if stats['durations']:
                stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
                stats['min_duration'] = min(stats['durations'])
                stats['max_duration'] = max(stats['durations'])
            if stats['memory_usage']:
                stats['avg_memory'] = sum(stats['memory_usage']) / len(stats['memory_usage'])
                stats['peak_memory'] = max(stats['memory_usage'])
            
            # Remove raw lists for cleaner output
            del stats['durations']
            del stats['memory_usage']
        
        return {
            "status": "success",
            "operations": operation_stats,
            "total_operations": len(benchmark_collector.metrics),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get operation breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get operation breakdown: {str(e)}")


@router.post("/clear-data")
async def clear_benchmark_data() -> Dict[str, Any]:
    """Clear all benchmark data."""
    try:
        # Clear all metrics
        benchmark_collector.metrics.clear()
        benchmark_collector.active_operations.clear()
        benchmark_collector.session_metrics.clear()
        
        # Clear monitoring data
        performance_monitor.metrics_history.clear()
        
        logger.info("🧹 Cleared all benchmark data")
        
        return {
            "status": "success",
            "message": "All benchmark data cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to clear benchmark data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")


@router.get("/health")
async def benchmark_health_check() -> Dict[str, Any]:
    """Health check for benchmarking system."""
    try:
        return {
            "status": "healthy",
            "benchmark_collector_active": True,
            "performance_monitor_active": performance_monitor.monitoring,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Benchmark health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
