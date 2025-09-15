import json
import random
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# === CONFIGURATION ===
API_BASE_URL = "http://127.0.0.1:80"  # your FastAPI base
START_ENDPOINT = "/move/teleop/udp"
STOP_ENDPOINT = "/move/teleop/udp/stop"
BUFFER_SIZE = 4096
UDP_TIMEOUT = 2.0  # seconds

# Number of total messages to send
TOTAL_MESSAGES = 100
# Number of parallel workers
NUM_WORKERS = 50

# Stats storage
total_sent = 0
total_received = 0
latencies = []
action_counters = []
lock = threading.Lock()


def start_udp_server():
    resp = requests.post(API_BASE_URL + START_ENDPOINT)
    resp.raise_for_status()
    info = resp.json()
    host = info["host"]
    port = info["port"]
    print(f"[API] UDP server running on {host}:{port}")
    return host, port


def stop_udp_server():
    resp = requests.post(API_BASE_URL + STOP_ENDPOINT)
    resp.raise_for_status()
    print("[API] UDP server stopped")


def send_udp_and_receive(host: str, port: int, payload: dict):
    global total_sent, total_received
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(UDP_TIMEOUT)

    data = json.dumps(payload).encode("utf-8")
    start_time = time.perf_counter()

    try:
        sock.sendto(data, (host, port))
        with lock:
            total_sent += 1

        raw, _ = sock.recvfrom(BUFFER_SIZE)
        elapsed = time.perf_counter() - start_time
        with lock:
            latencies.append(elapsed)
            total_received += 1

        text = raw.decode("utf-8")
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            print(f"✖ Non-JSON response: {text}")
            return

        if "nb_actions_received" in resp:
            with lock:
                action_counters.append(resp["nb_actions_received"])
        else:
            print(f"→ {resp}")

    except socket.timeout:
        print(f"✖ Timeout for payload {payload}")
    finally:
        sock.close()


def generate_payload(index: int):
    # Randomize values and timestamp
    payload = {
        "x": 0,
        "y": 0,
        "z": 0,
        "rx": 0,
        "ry": 0,
        "rz": 0,
        # "rx": random.uniform(-10, 10),
        # "ry": random.uniform(-10, 10),
        # "rz": random.uniform(-10, 10),
        "open": random.choice([0.0, 1.0]),
        "source": random.choice(["right"]),
        "timestamp": time.time() + index * 0.001,
    }
    return payload


def main():
    host, port = start_udp_server()

    # Prepare randomized payload list
    payloads = [generate_payload(i) for i in range(TOTAL_MESSAGES)]
    # random.shuffle(payloads)

    print(f"Sending {TOTAL_MESSAGES} messages with {NUM_WORKERS} workers...")
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(send_udp_and_receive, host, port, p) for p in payloads
        ]
        # Optional: iterate to catch exceptions
        for f in as_completed(futures):
            _ = f.result()

    total_time = time.perf_counter() - start_all
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
    throughput = total_received / total_time if total_time > 0 else float("nan")
    avg_actions = sum(action_counters) / len(action_counters) if action_counters else 0

    print("\n=== PERFORMANCE REPORT ===")
    print(f"Total sent:     {total_sent}")
    print(f"Total received: {total_received}")
    print(f"Total time:     {total_time:.2f}s")
    print(f"Avg latency:    {avg_latency*1000:.2f}ms")
    print(f"Throughput:     {throughput:.2f} msgs/s")
    print(f"Avg actions received per report: {avg_actions:.2f}")

    stop_udp_server()


if __name__ == "__main__":
    main()
