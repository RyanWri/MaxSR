import torch
import platform
import psutil
import os


# Function to collect and return all system information as a string
def collect_system_info():
    info = []

    # GPU Information using PyTorch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (
                1024**3
            )  # in GB
            gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # in GB
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # in GB
            gpu_memory_free = gpu_memory_total - gpu_memory_allocated  # in GB

            info.append(f"GPU {i}: {gpu_name}")
            info.append(f"  - Total Memory: {gpu_memory_total:.2f} GB")
            info.append(f"  - Memory Reserved: {gpu_memory_reserved:.2f} GB")
            info.append(f"  - Memory Allocated: {gpu_memory_allocated:.2f} GB")
            info.append(f"  - Memory Free: {gpu_memory_free:.2f} GB")
    else:
        info.append("No GPU available.")

    # CPU Information
    info.append(f"\nCPU: {platform.processor()}")
    info.append(
        f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )
    info.append(f"Max Frequency: {psutil.cpu_freq().max:.2f} MHz")
    info.append(f"Current Frequency: {psutil.cpu_freq().current:.2f} MHz")
    info.append(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

    # Memory Information
    memory = psutil.virtual_memory()
    info.append(f"\nTotal Memory: {memory.total / (1024 ** 3):.2f} GB")
    info.append(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    info.append(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
    info.append(f"Memory Usage: {memory.percent}%")

    # Disk Information
    disk = psutil.disk_usage("/")
    info.append(f"\nTotal Disk Space: {disk.total / (1024 ** 3):.2f} GB")
    info.append(f"Used Disk Space: {disk.used / (1024 ** 3):.2f} GB")
    info.append(f"Free Disk Space: {disk.free / (1024 ** 3):.2f} GB")
    info.append(f"Disk Usage: {disk.percent}%")

    # Operating System Information
    info.append(
        f"\nOperating System: {platform.system()} {platform.release()} {platform.version()}"
    )
    info.append(f"Machine: {platform.machine()}")
    info.append(f"Node Name: {platform.node()}")
    info.append(f"Python Version: {platform.python_version()}")

    return "\n".join(info)


# Collect system information
system_info = collect_system_info()

# Save the information to a text file
file_path = "system_info.txt"
with open(file_path, "w") as file:
    file.write(system_info)

print(f"System information has been saved to {file_path}")
