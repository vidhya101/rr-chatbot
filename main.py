"""
Main App
--------
Starts all the microservices and provides a single entry point for the entire system
with enhanced monitoring, health checks, and service management.
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import signal
import json
import psutil
import requests
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Enhanced logging configuration
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "app_rotating.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger("main-app")

# Service status enum
class ServiceStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    CRASHED = "crashed"
    RESTARTING = "restarting"

# Service configuration dataclass
@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    port: int
    dependencies: List[str]
    cwd: Optional[str] = None
    health_check_endpoint: Optional[str] = None
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    environment: Dict[str, str] = None
    required_memory_mb: int = 512
    required_cpu_percent: float = 50.0
    auto_restart: bool = True
    startup_timeout: int = 30
    graceful_shutdown_timeout: int = 10

# Load service configurations from YAML
def load_service_configs(config_path: str = "services.yaml") -> Dict[str, ServiceConfig]:
    if not os.path.exists(config_path):
        # Default configurations
        return {
            "api_gateway": ServiceConfig(
                name="api_gateway",
                command=["python", "api_gateway.py"],
                port=8000,
                dependencies=[],
                health_check_endpoint="/health",
                environment={"PYTHONUNBUFFERED": "1"}
            ),
            "data_processing": ServiceConfig(
                name="data_processing",
                command=["python", "data_processing_service.py"],
                port=8001,
                dependencies=[],
                health_check_endpoint="/health",
                environment={"PYTHONUNBUFFERED": "1"}
            ),
            "ml": ServiceConfig(
                name="ml",
                command=["python", "ml_service.py"],
                port=8002,
                dependencies=[],
                health_check_endpoint="/health",
                environment={"PYTHONUNBUFFERED": "1"}
            ),
            "visualization": ServiceConfig(
                name="visualization",
                command=["python", "visualization_service.py"],
                port=8003,
                dependencies=[],
                health_check_endpoint="/health",
                environment={"PYTHONUNBUFFERED": "1"}
            ),
            "llm": ServiceConfig(
                name="llm",
                command=["python", "llm_service.py"],
                port=8004,
                dependencies=[],
                health_check_endpoint="/health",
                environment={"PYTHONUNBUFFERED": "1"}
            ),
            "frontend": ServiceConfig(
                name="frontend",
                command=["npm", "start"],
                port=3000,
                dependencies=["api_gateway"],
                cwd="./frontend",
                health_check_endpoint="/",
                environment={"NODE_ENV": "development"}
            )
        }
    
    with open(config_path) as f:
        configs = yaml.safe_load(f)
        return {
            name: ServiceConfig(**config) 
            for name, config in configs.items()
        }

# Enhanced service manager
class ServiceManager:
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = load_service_configs()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status: Dict[str, ServiceStatus] = {}
        self.metrics: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.health_check_threads: Dict[str, threading.Thread] = {}
        self.stop_event = threading.Event()

    def check_system_resources(self, service: ServiceConfig) -> bool:
        """Check if system has enough resources to start the service"""
        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            cpu_percent = psutil.cpu_percent()

            if available_memory < service.required_memory_mb:
                logger.warning(f"Not enough memory to start {service.name}. Required: {service.required_memory_mb}MB, Available: {available_memory:.2f}MB")
                return False

            if cpu_percent > service.required_cpu_percent:
                logger.warning(f"CPU usage too high to start {service.name}. Maximum allowed: {service.required_cpu_percent}%, Current: {cpu_percent:.2f}%")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return True  # Continue anyway if resource check fails

    def health_check(self, service_name: str):
        """Perform health checks for a service"""
        service = self.services[service_name]
        if not service.health_check_endpoint:
            return

        while not self.stop_event.is_set():
            try:
                url = f"http://localhost:{service.port}{service.health_check_endpoint}"
                response = requests.get(url, timeout=5)
                
                with self.lock:
                    if response.status_code == 200:
                        self.status[service_name] = ServiceStatus.RUNNING
                        self.metrics[service_name] = {
                            "last_health_check": datetime.now(),
                            "status_code": response.status_code,
                            "response_time": response.elapsed.total_seconds()
                        }
                    else:
                        logger.warning(f"Health check failed for {service_name}: {response.status_code}")
                        if self.status[service_name] == ServiceStatus.RUNNING:
                            self.status[service_name] = ServiceStatus.FAILED
            except Exception as e:
                logger.error(f"Health check error for {service_name}: {str(e)}")
                with self.lock:
                    if self.status[service_name] == ServiceStatus.RUNNING:
                        self.status[service_name] = ServiceStatus.FAILED

            time.sleep(service.health_check_interval)

    def start_service(self, service_name: str) -> bool:
        """Start a service with enhanced error handling and monitoring"""
        service = self.services[service_name]
        
        # Check dependencies
        for dependency in service.dependencies:
            if (dependency not in self.processes or 
                self.status.get(dependency) not in {ServiceStatus.RUNNING, ServiceStatus.STARTING}):
                logger.error(f"Cannot start {service_name}: dependency {dependency} is not running")
                return False

        # Check system resources
        if not self.check_system_resources(service):
            return False

        try:
            logger.info(f"Starting {service_name}...")
            
            with self.lock:
                self.status[service_name] = ServiceStatus.STARTING
            
            # Prepare environment variables
            env = os.environ.copy()
            if service.environment:
                env.update(service.environment)

            # Start the process
            process = subprocess.Popen(
                service.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=service.cwd,
                env=env
            )
            
            # Wait for startup
            start_time = time.time()
            while time.time() - start_time < service.startup_timeout:
                if process.poll() is not None:
                    logger.error(f"Service {service_name} failed to start")
                    with self.lock:
                        self.status[service_name] = ServiceStatus.FAILED
                    return False
                
                # Try health check
                try:
                    if service.health_check_endpoint:
                        url = f"http://localhost:{service.port}{service.health_check_endpoint}"
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            break
                except:
                    pass
                
                time.sleep(1)
            
            with self.lock:
                self.processes[service_name] = process
                self.status[service_name] = ServiceStatus.RUNNING
                
                # Start health check thread
                if service.health_check_endpoint:
                    thread = threading.Thread(
                        target=self.health_check,
                        args=(service_name,),
                        daemon=True
                    )
                    self.health_check_threads[service_name] = thread
                    thread.start()
            
            logger.info(f"Service {service_name} started with PID {process.pid}")
            return True
        
        except Exception as e:
            logger.error(f"Error starting {service_name}: {str(e)}")
            with self.lock:
                self.status[service_name] = ServiceStatus.FAILED
            return False

    def stop_service(self, service_name: str):
        """Stop a service with graceful shutdown"""
        service = self.services[service_name]
        
        if service_name not in self.processes:
            return

        process = self.processes[service_name]
        if process.poll() is None:
            logger.info(f"Stopping {service_name}...")
            try:
                # Try graceful shutdown first
                process.terminate()
                try:
                    process.wait(timeout=service.graceful_shutdown_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Service {service_name} did not terminate gracefully, forcing...")
                    process.kill()
                    process.wait()
                
                logger.info(f"Service {service_name} stopped")
                
                with self.lock:
                    self.status[service_name] = ServiceStatus.STOPPED
                    if service_name in self.health_check_threads:
                        self.health_check_threads[service_name].join(timeout=1)
                        del self.health_check_threads[service_name]
            
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {str(e)}")

    def monitor_output(self, service_name: str):
        """Monitor service output with enhanced logging"""
        process = self.processes[service_name]
        service = self.services[service_name]

        def log_output(pipe, level):
            for line in iter(pipe.readline, ''):
                if line:
                    if level == logging.INFO:
                        logger.info(f"[{service_name}] {line.strip()}")
                    else:
                        logger.error(f"[{service_name}] {line.strip()}")

        # Monitor stdout and stderr in separate threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(log_output, process.stdout, logging.INFO)
            executor.submit(log_output, process.stderr, logging.ERROR)

    def get_service_metrics(self, service_name: str) -> Dict:
        """Get detailed metrics for a service"""
        if service_name not in self.processes:
            return {}

        try:
            process = psutil.Process(self.processes[service_name].pid)
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "status": self.status[service_name].value,
                "uptime": time.time() - process.create_time(),
                "threads": len(process.threads()),
                "last_health_check": self.metrics.get(service_name, {}).get("last_health_check"),
                "health_check_response_time": self.metrics.get(service_name, {}).get("response_time")
            }
        except Exception as e:
            logger.error(f"Error getting metrics for {service_name}: {str(e)}")
            return {}

    def start_all_services(self, specific_services: Optional[List[str]] = None):
        """Start services with dependency resolution"""
        services_to_start = specific_services or list(self.services.keys())
        
        # Create dependency graph
        dependency_graph = {
            service: self.services[service].dependencies 
            for service in services_to_start
        }
        
        # Resolve dependencies
        start_order = []
        while dependency_graph:
            # Find services with no dependencies
            ready = [s for s, deps in dependency_graph.items() if not deps]
            
            if not ready:
                logger.error("Circular dependency detected")
                break
            
            start_order.extend(ready)
            
            # Remove started services from graph
            for service in ready:
                dependency_graph.pop(service)
            
            # Remove started services from dependencies
            for deps in dependency_graph.values():
                for service in ready:
                    if service in deps:
                        deps.remove(service)
        
        # Start services in order
        with ThreadPoolExecutor(max_workers=len(self.services)) as executor:
            for service_name in start_order:
                if self.start_service(service_name):
                    executor.submit(self.monitor_output, service_name)

    def stop_all_services(self):
        """Stop all services in reverse dependency order"""
        self.stop_event.set()
        
        # Stop in reverse dependency order
        for service_name in reversed(list(self.processes.keys())):
            self.stop_service(service_name)

# Enhanced signal handler
def handle_signal(sig, frame):
    """Handle signals with graceful shutdown"""
    logger.info("Initiating graceful shutdown...")
    service_manager.stop_all_services()
    sys.exit(0)

# Create service manager instance
service_manager = ServiceManager()

def main():
    """Enhanced main entry point with CLI options"""
    parser = argparse.ArgumentParser(
        description="Enhanced Data Visualization Chatbot Service Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--services",
        nargs="+",
        help="Specific services to start"
    )
    parser.add_argument(
        "--config",
        help="Path to service configuration YAML file",
        default="services.yaml"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        # Start services
        service_manager.start_all_services(args.services)
        
        # Main monitoring loop
        while True:
            # Check service health
            for service_name, process in list(service_manager.processes.items()):
                service = service_manager.services[service_name]
                
                if process.poll() is not None:
                    logger.error(f"Service {service_name} has crashed with exit code {process.poll()}")
                    
                    with service_manager.lock:
                        service_manager.status[service_name] = ServiceStatus.CRASHED
                    
                    # Auto-restart if configured
                    if service.auto_restart:
                        logger.info(f"Attempting to restart {service_name}...")
                        if service_manager.start_service(service_name):
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                executor.submit(service_manager.monitor_output, service_name)
                
                # Log metrics
                metrics = service_manager.get_service_metrics(service_name)
                if metrics:
                    logger.debug(f"Service {service_name} metrics: {json.dumps(metrics, default=str)}")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        service_manager.stop_all_services()

if __name__ == "__main__":
    main() 