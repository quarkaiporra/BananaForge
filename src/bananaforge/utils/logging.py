"""Logging configuration for BananaForge."""

import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_colors: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration for BananaForge.

    Args:
        level: Logging level
        log_file: Optional log file path
        enable_colors: Whether to use colored output
        format_string: Custom format string
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Default format
    if format_string is None:
        if enable_colors and sys.stderr.isatty():
            format_string = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s"
        else:
            format_string = "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    if enable_colors and sys.stderr.isatty():
        # Colored formatter
        formatter = colorlog.ColoredFormatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)

    # File handler if specified
    handlers = [console_handler]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file

        # File formatter (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Set specific logger levels
    _configure_library_loggers()


def _configure_library_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Set BananaForge loggers to appropriate levels
    logging.getLogger("bananaforge").setLevel(logging.INFO)


class ProgressLogger:
    """Logger for tracking optimization progress."""

    def __init__(self, name: str = "bananaforge.progress"):
        """Initialize progress logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.last_step = 0

    def log_optimization_step(
        self, step: int, total_steps: int, loss_dict: dict, frequency: int = 10
    ) -> None:
        """Log optimization step progress.

        Args:
            step: Current step
            total_steps: Total number of steps
            loss_dict: Dictionary of loss values
            frequency: Logging frequency (every N steps)
        """
        if step % frequency == 0 or step == total_steps - 1:
            progress_pct = (step / total_steps) * 100
            total_loss = loss_dict.get("total", 0)

            if isinstance(total_loss, (int, float)):
                loss_str = f"{total_loss:.4f}"
            else:
                # Handle tensor values
                loss_str = (
                    f"{total_loss.item():.4f}"
                    if hasattr(total_loss, "item")
                    else str(total_loss)
                )

            self.logger.info(
                f"Step {step:4d}/{total_steps} ({progress_pct:5.1f}%) - Loss: {loss_str}"
            )

            # Log detailed losses at debug level
            if self.logger.isEnabledFor(logging.DEBUG):
                loss_details = []
                for key, value in loss_dict.items():
                    if key != "total":
                        if hasattr(value, "item"):
                            loss_details.append(f"{key}: {value.item():.4f}")
                        else:
                            loss_details.append(f"{key}: {value:.4f}")

                if loss_details:
                    self.logger.debug(f"  Detailed losses: {', '.join(loss_details)}")

    def log_material_selection(self, materials: list, colors: list) -> None:
        """Log selected materials.

        Args:
            materials: List of material names/IDs
            colors: List of color values
        """
        self.logger.info(f"Selected {len(materials)} materials:")
        for i, (material, color) in enumerate(zip(materials, colors)):
            if hasattr(color, "cpu"):
                color_vals = color.cpu().numpy()
                color_str = f"RGB({color_vals[0]:.2f}, {color_vals[1]:.2f}, {color_vals[2]:.2f})"
            else:
                color_str = str(color)
            self.logger.info(f"  {i+1}. {material} - {color_str}")

    def log_export_results(self, output_files: dict) -> None:
        """Log export results.

        Args:
            output_files: Dictionary of output file paths
        """
        self.logger.info("Export completed successfully:")
        for file_type, file_path in output_files.items():
            self.logger.info(f"  {file_type}: {Path(file_path).name}")


class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, name: str = "bananaforge.performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(name)
        self.timers = {}

    def start_timer(self, name: str) -> None:
        """Start a performance timer.

        Args:
            name: Timer name
        """
        import time

        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a performance timer and log result.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds
        """
        import time

        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0

        elapsed = time.time() - self.timers[name]
        del self.timers[name]

        self.logger.debug(f"{name}: {elapsed:.2f}s")
        return elapsed

    def log_memory_usage(self, step: str = "current") -> None:
        """Log current memory usage.

        Args:
            step: Description of current step
        """
        try:
            import psutil
            import torch

            # System memory
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            memory_pct = memory.percent

            # GPU memory if available
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_total = torch.cuda.max_memory_allocated() / (1024**3)
                gpu_info = f", GPU: {gpu_memory:.1f}/{gpu_total:.1f}GB"

            self.logger.debug(
                f"Memory usage at {step}: {memory_gb:.1f}GB ({memory_pct:.1f}%){gpu_info}"
            )

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.warning(f"Could not log memory usage: {e}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info() -> None:
    """Log system information for debugging."""
    logger = get_logger("bananaforge.system")

    import platform
    import torch

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")

    # GPU information
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"  GPU Memory: {torch.cuda.max_memory_allocated() / (1024**3):.1f}GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("  MPS (Apple Metal) available")
    else:
        logger.info("  GPU: Not available (using CPU)")

    # Memory information
    try:
        import psutil

        memory = psutil.virtual_memory()
        logger.info(f"  RAM: {memory.total / (1024**3):.1f}GB total")
    except ImportError:
        pass
