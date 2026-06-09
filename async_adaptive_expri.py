import os
import torch
import torch.distributed as dist

def run_adaptive_profiling():
    device = torch.device(f"cuda:{dist.get_rank()}")
    torch.cuda.set_device(device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print("          ADAPTIVE THRESHOLD PROFILING (DIRECTION D)")
        print("=" * 80)
        print(f"GPUs: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print("-" * 80)
        print(f"{'Batch':<6} | {'Seq Len':<8} | {'Total Toks':<10} | {'TP Comm (ms)':<15} | {'SP Comm (ms)':<15} | {'Better Mode':<12}")
        print("-" * 80)

    # Configs
    hidden_size = 4096
    
    # We test different batch sizes and sequence lengths to see the crossover point
    batch_sizes = [1, 4, 16, 32]
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    
    # CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    dummy_tensor = torch.randn((1024 * 4, hidden_size), device=device, dtype=torch.float16)
    dummy_tensor_out = torch.empty_like(dummy_tensor)
    for _ in range(10):
        dist.all_reduce(dummy_tensor)
        dist.all_to_all_single(dummy_tensor_out, dummy_tensor)
    torch.cuda.synchronize()
    dist.barrier()
    
    for bs in batch_sizes:
        for seq_len in seq_lens:
            total_tokens = bs * seq_len
            
            # 1. TP Communication Simulation (2 * All-Reduce per layer)
            # Size of tensor: [total_tokens, hidden_size]
            tp_tensor = torch.randn((total_tokens, hidden_size), device=device, dtype=torch.float16)
            
            tp_times = []
            for _ in range(10):
                start_event.record()
                # Simulate 2 All-Reduces representing 1 transformer layer
                dist.all_reduce(tp_tensor)
                dist.all_reduce(tp_tensor)
                end_event.record()
                torch.cuda.synchronize()
                tp_times.append(start_event.elapsed_time(end_event))
            
            avg_tp_comm = sum(tp_times) / 10
            
            # 2. SP Communication Simulation (2 * All-To-All per layer)
            # Size of tensor per GPU: [total_tokens // world_size, hidden_size]
            # Ulysses performs all-to-all where each rank sends a chunk of size [total_tokens // world_size, hidden_size // world_size] to every other rank.
            # Total size sent/received per rank: total_tokens * hidden_size / world_size
            sp_tensor_in = torch.randn((world_size, total_tokens // world_size, hidden_size // world_size), device=device, dtype=torch.float16)
            sp_tensor_out = torch.empty_like(sp_tensor_in)
            
            sp_times = []
            for _ in range(10):
                start_event.record()
                # Simulate 2 All-to-Alls representing 1 transformer layer
                dist.all_to_all_single(sp_tensor_out, sp_tensor_in)
                dist.all_to_all_single(sp_tensor_in, sp_tensor_out)
                end_event.record()
                torch.cuda.synchronize()
                sp_times.append(start_event.elapsed_time(end_event))
                
            avg_sp_comm = sum(sp_times) / 10
            
            better_mode = "SP" if avg_sp_comm < avg_tp_comm else "TP"
            
            if rank == 0:
                print(f"{bs:<6} | {seq_len:<8} | {total_tokens:<10} | {avg_tp_comm:<15.3f} | {avg_sp_comm:<15.3f} | {better_mode:<12}")
                
    if rank == 0:
        print("=" * 80)
        print("CONCLUSION:")
        print("Notice how the Crossover Point (where SP becomes faster than TP) is NOT a simple constant threshold.")
        print("It depends on the balance between All-Reduce bandwidth limits (dense ring/tree) and All-to-All latency/bandwidth.")
        print("An adaptive threshold scheduler should dynamically choose the optimal parallelism mode based on (Batch Size, Sequence Length).")
        print("=" * 80)

if __name__ == "__main__":
    dist.init_process_group("nccl")
    run_adaptive_profiling()
    dist.destroy_process_group()
