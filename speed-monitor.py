#p95程序
import csv
import os
import socket
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import keyboard  # Install with `pip install keyboard`
import numpy as np  # Install with `pip install numpy`

# Suppress HTTPS warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class DownloadMonitor:
    """Monitor download progress and statistics."""
    def __init__(self, url, total_size, num_threads):
        self.url = url
        self.total_size = total_size
        self.downloaded_size = 0
        self.size_buffer = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.num_threads = num_threads
        self.instantaneous_speeds = []  # List to store instantaneous speeds
        self.stop_requested = False
        self.last_update = self.start_time

    def update(self, chunk_size, elapsed_time):
        """Update progress and calculate speeds."""
        with self.lock:
            self.size_buffer += chunk_size
            self.downloaded_size += chunk_size
            if time.time() - self.last_update >= 0.1:
                instantaneous_speed = self.size_buffer / (time.time() - self.last_update)
                self.instantaneous_speeds.append(instantaneous_speed)
                self.last_update = time.time()
                self.size_buffer = 0
    def finish(self):
        with self.lock:
            instantaneous_speed = self.size_buffer / (time.time() - self.last_update)
            self.instantaneous_speeds.append(instantaneous_speed)
            self.last_update = time.time()
            self.size_buffer = 0

    def get_pct_speed(self, pct):
        """Calculate the 95th percentile of instantaneous speeds."""
        if self.instantaneous_speeds:
            return np.percentile(self.instantaneous_speeds, pct)
        return 0
    
    def get_avg_speed(self):
        """Calculate the average download speed."""
        elapsed_time = self.last_update - self.start_time
        return self.downloaded_size / elapsed_time if elapsed_time > 0 else 0

    def request_stop(self):
        """Request to stop the download."""
        self.stop_requested = True

    def should_stop(self):
        """Check if a stop has been requested."""
        return self.stop_requested

    def display_progress(self):
        """Display real-time download statistics."""
        elapsed_time = time.time() - self.start_time
        average_speed = self.downloaded_size / elapsed_time if elapsed_time > 0 else 0
        progress = self.downloaded_size / self.total_size * 100
        inst_speed = self.instantaneous_speeds[-1] if self.instantaneous_speeds else 0
        p50_speed = self.get_pct_speed(50)
        p95_speed = self.get_pct_speed(95)
        print(
            f"\rProgress: {progress:.2f}% | "
            f"Avg: {average_speed / 1024 / 1024 * 8:.2f} Mb/s | "
            f"Inst: {inst_speed / 1024 / 1024 * 8:.2f} Mb/s | "
            f"P50: {p50_speed / 1024 / 1024 * 8:.2f} Mb/s | "
            f"P95: {p95_speed / 1024 / 1024 * 8:.2f} Mb/s",
            end="        ",
        )

def download_chunk(url, start, end, chunk_index, output_dir, monitor):
    """Download a specific chunk of the file and update monitor."""
    headers = {
        "Referer": "https://mirrors.zju.edu.cn",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True, verify=False)
    output_path = os.path.join(output_dir, f"chunk_{chunk_index}")

    with open(output_path, "wb") as f:
        last_time = time.time()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                if monitor.should_stop():
                    return  # Stop download if requested
                f.write(chunk)
                current_time = time.time()
                monitor.update(len(chunk), current_time - last_time)
                last_time = current_time
        monitor.finish()
    return output_path

def merge_chunks(output_dir, output_file, total_chunks):
    """Merge all chunks into a single file."""
    if output_file is not None:
        with open(output_file, "wb") as outfile:
            for i in range(total_chunks):
                with open(os.path.join(output_dir, f"chunk_{i}"), "rb") as infile:
                    outfile.write(infile.read())
    for i in range(total_chunks):
        os.remove(os.path.join(output_dir, f"chunk_{i}"))


def multithreaded_download(url, output_file):
    """Download a file using multiple threads."""
    response = requests.head(url, verify=False)
    total_size = int(response.headers.get("content-length", 0))
    if total_size == 0:
        raise ValueError("Unable to determine file size.")

    cpu_count = min(multiprocessing.cpu_count(), 8)  # Number of threads = CPU cores
    chunk_size = total_size // cpu_count
    output_dir = "temp_chunks"
    os.makedirs(output_dir, exist_ok=True)

    print(f"File size: {total_size / (1024 * 1024):.2f} MB")

    monitor = DownloadMonitor(url, total_size, cpu_count)

    # Start a background thread to monitor progress every 0.2 seconds
    def monitor_progress():
        while not monitor.should_stop():
            monitor.display_progress()
            time.sleep(0.2)

        

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    # Start listening for 'q' key to stop the download
    def listen_for_quit():
        keyboard.wait("q")
        monitor.request_stop()

    keyboard_thread = threading.Thread(target=listen_for_quit, daemon=True)
    keyboard_thread.start()

    thread_count = cpu_count if total_size > 3 * 1024 * 1024 else 1  # Use single thread for small files
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        for i in range(thread_count):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < thread_count - 1 else total_size - 1
            futures.append(executor.submit(download_chunk, url, start, end, i, output_dir, monitor))

        for future in futures:
            if monitor.should_stop():
                break
            future.result()  # Wait for all threads to complete
    monitor.request_stop()
    merge_chunks(output_dir, output_file, thread_count)
    print(f"\rsize: {monitor.downloaded_size / (1024 * 1024):.2f} MB |"
       f"time: {monitor.last_update - monitor.start_time:.2f} seconds | "
       f"avg speed: {monitor.get_avg_speed() / (1024 * 1024) * 8:.2f} MB/s | "
       f"P50 speed: {monitor.get_pct_speed(50) / 1024 / 1024 * 8:.2f} MB/s | "
       f"P95 speed: {monitor.get_pct_speed(95) / 1024 / 1024 * 8:.2f} MB/s | ",
    )
    write_header = not os.path.exists("raw_data.csv")
    try:
        with open("raw_data.csv", mode='a', newline='') as logfile:
            writer = csv.writer(logfile)
            if write_header:
                writer.writerow(["设备名", "开始时间", "url", "文件大小", "持续时间", "线程数", "平均速度", "P50速度", "P95速度", "打点数据"])
            writer.writerow(["e5-2640v4", (monitor.start_time + 28800) / 86400 + 25569 , url, total_size,
                            monitor.last_update - monitor.start_time, thread_count, monitor.get_avg_speed(), 
                            monitor.get_pct_speed(50), monitor.get_pct_speed(95), monitor.instantaneous_speeds])
    except Exception as e:
        print(f"Error: {e}")
        
    write_header = not os.path.exists("speed_log.csv")
    try:
        with open("speed_log.csv", mode='a', newline='') as logfile:
            writer = csv.writer(logfile)
            if write_header:
                writer.writerow(["设备名", "开始时间", "url", "文件大小(MB)", "持续时间(s)", "线程数", "平均速度", "P50速度", "P95速度", "打点数据"])
            writer.writerow([socket.gethostname(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(monitor.start_time)), url, format_filesize(total_size),
                    round(monitor.last_update - monitor.start_time, 2), thread_count, format_filesize(monitor.get_avg_speed()*8), 
                    format_filesize(monitor.get_pct_speed(50)*8), format_filesize(monitor.get_pct_speed(95)*8), monitor.instantaneous_speeds])
    except Exception as e:
        print(f"Error: {e}")
def format_filesize(size_in_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_in_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


if __name__ == "__main__":
    urls = [
        "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png", #15kb
        "https://img.alicdn.com/imgextra/i4/O1CN01Dc9PxV1XzlCMO1boO_!!6000000002995-2-tps-3456-864.png", #300kb
        "https://img.alicdn.com/imgextra/i2/O1CN01zPdoog1hSSbHWDV6K_!!6000000004276-1-tps-960-288.gif", #3mb
        "https://dldir1.qq.com/qqfile/qq/QQNT/Windows/QQ_9.9.17_241213_x64_01.exe", # 200mb
        "https://download.ca.newthread.run/Japan%20Kyoto%20Cherry%20Blossom%20%E4%BA%AC%E9%83%BD%E3%81%AE%E6%A1%9C%20%5B4zgydqY9dTo%5D.mp4", # 200mb
        #"https://wirelesscdn-download.xuexi.cn/publish/xuexi_android/latest/xuexi_android_10002068.apk"
    ]
    #url = "https://mirrors.zju.edu.cn/opensuse/tumbleweed/iso/openSUSE-Tumbleweed-GNOME-Live-x86_64-Current.iso"
    #url = "https://node-36-250-1-90.speedtest.cn:51090/download?size=100000000"
    #url = "https://wirelesscdn-download.xuexi.cn/publish/xuexi_android/latest/xuexi_android_10002068.apk"
    for url in urls:
        output_file = url.split("/")[-1]
        output_file = None
        try:
            multithreaded_download(url, output_file=output_file)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
        print("\n")
    exit(0)