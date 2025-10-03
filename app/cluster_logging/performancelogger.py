import streamlit as st
import time
import psutil
import platform
from datetime import datetime

class PerformanceLogger:
    def __init__(self):
        self.logs = []
        self.start_time = None
        self.peak_memory = 0
        self.peak_cpu = 0

    def start(self):
        self.start_time = time.time()
        self.log("Process started")

    def log(self, message, **kwargs):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            **kwargs
        }
        self.logs.append(entry)

    def update_peak_memory(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        current_mem = mem_info.rss / (1024 ** 2)
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem

    def update_peak_cpu(self):
        current_cpu = psutil.cpu_percent(interval=0.1)
        if current_cpu > self.peak_cpu:
            self.peak_cpu = current_cpu

    def get_summary(self, num_texts, num_clusters):
        total_time = time.time() - self.start_time
        return {
            "total_processing_time": f"{total_time:.2f} seconds",
            "peak_memory_usage": f"{self.peak_memory:.2f} MB",
            "num_texts_processed": f"{num_texts:,}",
            "num_clusters_formed": f"{num_clusters:,}",
            "peak_cpu_usage": f"{self.peak_cpu:.1f}%",
            "system_info": f"{platform.system()} {platform.release()}, {psutil.cpu_count()} cores"
        }

    def store_summary(self, num_texts, num_clusters):
        st.session_state.performance_summary = self.get_summary(num_texts, num_clusters)
