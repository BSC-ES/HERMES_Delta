from .constants import precision

__all__ = ["ConfigDelta", "configure_logger", "log_message",
           "get_time_stamp", "get_memory_stamp", "record_time", "record_memory",
           "finalize_time_log", "finalize_mem_log", "precision"]


def __getattr__(name):
    if name == "ConfigDelta":
        from .config import ConfigDelta
        return ConfigDelta

    logger_names = [
        "configure_logger", "log_message", "get_time_stamp", "get_memory_stamp", "record_time", "record_memory",
        "finalize_time_log", "finalize_mem_log",
    ]
    if name in logger_names:
        from . import logger_config
        return getattr(logger_config, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
