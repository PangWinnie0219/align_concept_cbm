import json
import threading
import atexit
import signal


class Logger:
    """
    A class for logging data to a file with options for multithreading and forced logging. 
    1) Incremental Saving: Logger saves logs efficiently by accumulating them in memory until memory or log count limits are reached, minimizing disk I/O.
    2) Delayed Disk I/O: Logger writes logs to the file system only when necessary, avoiding frequent disk operations and reducing overhead.
    3) Non-blocking (with multi-threading): Logger operates without blocking the main script, ensuring efficient execution without slowdowns due to disk I/O.

    Args:
        log_file_path (str): The path to the log file.
        max_memory_limit (int, optional): The maximum memory limit for accumulated logs in bytes (default is 10MB).
        max_log_count (int, optional): The maximum number of logs to accumulate before saving (default is 1000).
        multithreading (bool, optional): Enable multithreading for background log saving (default is True).

    Attributes:
        log_file_path (str): The path to the log file.
        max_memory_limit (int): The maximum memory limit for accumulated logs in bytes.
        max_log_count (int): The maximum number of logs to accumulate before saving.
        accumulated_logs (list): A list to accumulate log entries.
        log_lock (threading.Lock or None): A lock for thread-safe log accumulation (None if multithreading is disabled).
        log_thread (threading.Thread or None): The thread for background log saving (None if multithreading is disabled).
        force_logging (bool): Indicates whether forced logging is enabled.
        multithreading (bool): Indicates whether multithreading is enabled.

    Methods:
        log(log_dict, force=False):
            Logs a dictionary log entry. If `force` is True, the log will be saved immediately.

        join():
            Waits for the log-saving thread to finish (if multithreading is enabled).

    Example:
        # Create a logger with multithreading enabled
        logger = Logger("training_logs.json", max_memory_limit=2 * 1024 * 1024, max_log_count=1000, multithreading=True)

        # Log a message
        logger.log({"msg": "Training started"})
        logger.log({"loss": 0.1, "acc": 0.9, "epoch": 1}, force=True)

        # Force log saving
        logger.log({"message": "Force saving logs"}, force=True)

    This logger is coded by ChatGPT. There may be some bugs.
    """

    def __init__(self, log_file_path, max_memory_limit=10 * 1024 * 1024, max_log_count=1000, multithreading=True):
        self.log_file_path = log_file_path
        print(f"Logger will accumulate logs to {self.log_file_path} when max_memory_limit : {max_memory_limit / 1024 / 1024}MB or max_log_count: {max_log_count} is reached.")
        self.max_memory_limit = max_memory_limit
        self.max_log_count = max_log_count
        self.accumulated_logs = []
        self.log_lock = threading.Lock() if multithreading else None
        self.multithreading = multithreading

        atexit.register(self._exit_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if signum == signal.SIGINT:
            self._exit_handler()
            exit(1)

    def log(self, log_dict, force=False):
        if self.multithreading:
            with self.log_lock:
                self.accumulated_logs.append(log_dict)
        else:
            self.accumulated_logs.append(log_dict)

        if self._should_save_logs() or force:
            self._save_logs()

    def _should_save_logs(self):
        return len(self.accumulated_logs) >= self.max_log_count or self._get_accumulated_logs_size() >= self.max_memory_limit

    def _save_logs(self):
        if self.multithreading:
            with self.log_lock:
                logs_to_save = self.accumulated_logs.copy()
                self.accumulated_logs = []
        else:
            logs_to_save = self.accumulated_logs.copy()
            self.accumulated_logs = []

        if self.multithreading:
            save_thread = threading.Thread(target=self._save_logs_to_file, args=(logs_to_save,))
            save_thread.start()
        else:
            self._save_logs_to_file(logs_to_save)

    def _get_accumulated_logs_size(self):
        return sum(len(json.dumps(log)) for log in self.accumulated_logs)

    def _save_logs_to_file(self, logs_to_save):
        if not logs_to_save:
            return

        with open(self.log_file_path, "a") as log_file:
            for log in logs_to_save:
                log_line = json.dumps(log) + "\n"
                log_file.write(log_line)

        print(f"Saved {len(logs_to_save)} logs to {self.log_file_path}")

    def _exit_handler(self):
        if self.accumulated_logs:
            self._save_logs_to_file(self.accumulated_logs)
            self.accumulated_logs = []