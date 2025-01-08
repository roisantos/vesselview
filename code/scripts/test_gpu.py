import torch

def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available!")
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda_availability()
