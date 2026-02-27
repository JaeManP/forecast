import multiprocessing

# Cap at 16 to avoid memory exhaustion with large datasets
CPU_COUNT = min(multiprocessing.cpu_count(), 16)
DUMMY_TOKEN = "DUMMY"
LOCAL_DIR = "./artifacts/"