import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

def run_profiling():
    device = torch.device(f"cuda:{dist.get_rank()}")
    torch.cuda.set_device(device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print("          ULYSSES SP COMMUNICATION PROFILING (C1 DIRECTION)")
        print("=" * 80)
        print(f"GPUs (SP Size): {world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print("-" * 80)
        print(f"{'Seq Len':<10} | {'All-to-All 1 (ms)':<18} | {'Attention (ms)':<15} | {'All-to-All 2 (ms)':<18} | {'Total Comm %':<12}")
        print("-" * 80)

    # Configs
    num_heads = 32
    head_size = 128
    num_kv_heads = 8  # GQA
    total_head_dim = (num_heads + 2 * num_kv_heads) * head_size
    
    # We profile across different sequence lengths
    seq_lens = [2048, 4096, 8192, 16384]
    
    for seq_len in seq_lens:
        # Ulysses partitions along sequence dimension:
        seq_len_per_gpu = seq_len // world_size
        
        # Shapes before all-to-all 1:
        # Q: [seq_len_per_gpu, num_heads, head_size] -> flattened to [seq_len_per_gpu, num_heads * head_size]
        # K, V: [seq_len_per_gpu, num_kv_heads, head_size]
        q_shape = (seq_len_per_gpu, num_heads * head_size)
        kv_shape = (seq_len_per_gpu, num_kv_heads * head_size)
        
        q = torch.randn(q_shape, device=device, dtype=torch.float16)
        k = torch.randn(kv_shape, device=device, dtype=torch.float16)
        v = torch.randn(kv_shape, device=device, dtype=torch.float16)
        
        # Pack Q, K, V for all-to-all
        qkv = torch.cat([q, k, v], dim=-1)
        # Reshape to partition along head dimension for all-to-all
        # We need it to be partitioned into world_size chunks
        # Shape: [world_size, seq_len_per_gpu, (num_heads + 2*num_kv_heads)*head_size // world_size]
        assert total_head_dim % world_size == 0, "Ulysses requires total heads dim to be divisible by SP size"
        
        qkv = qkv.view(seq_len_per_gpu, world_size, total_head_dim // world_size).transpose(0, 1).contiguous()
        qkv_out = torch.empty_like(qkv)
        
        # CUDA Events for accurate timing
        start_event_a2a1 = torch.cuda.Event(enable_timing=True)
        end_event_a2a1 = torch.cuda.Event(enable_timing=True)
        
        start_event_attn = torch.cuda.Event(enable_timing=True)
        end_event_attn = torch.cuda.Event(enable_timing=True)
        
        start_event_a2a2 = torch.cuda.Event(enable_timing=True)
        end_event_a2a2 = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(5):
            dist.all_to_all_single(qkv_out, qkv)
            # Dummy attention on unpacked tensors
            q_, k_, v_ = qkv_out.split([
                (num_heads * head_size) // world_size,
                (num_kv_heads * head_size) // world_size,
                (num_kv_heads * head_size) // world_size
            ], dim=-1)
            # Reshape for SDPA: [batch_size=num_heads_per_gpu, seq_len, head_size]
            q_attn = q_.transpose(0, 1).reshape(num_heads // world_size, seq_len, head_size).contiguous()
            k_attn = k_.transpose(0, 1).reshape(num_kv_heads // world_size, seq_len, head_size).contiguous()
            v_attn = v_.transpose(0, 1).reshape(num_kv_heads // world_size, seq_len, head_size).contiguous()
            # Repeat KV for GQA if needed (SDPA handles GQA if batch sizes match, so we repeat k and v)
            k_attn = k_attn.repeat_interleave(num_heads // num_kv_heads, dim=0)
            v_attn = v_attn.repeat_interleave(num_heads // num_kv_heads, dim=0)
            
            c_ = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            # Pack c_ back
            c_ = c_.view(num_heads // world_size * head_size, seq_len).transpose(0, 1).contiguous()
            c_out = torch.empty_like(c_)
            dist.all_to_all_single(c_out, c_)
            
        torch.cuda.synchronize()
        dist.barrier()
        
        # Actual profiling runs
        steps = 20
        a2a1_times = []
        attn_times = []
        a2a2_times = []
        
        for _ in range(steps):
            # 1. Profile All-to-All 1
            start_event_a2a1.record()
            dist.all_to_all_single(qkv_out, qkv)
            end_event_a2a1.record()
            
            # Unpack
            q_, k_, v_ = qkv_out.split([
                (num_heads * head_size) // world_size,
                (num_kv_heads * head_size) // world_size,
                (num_kv_heads * head_size) // world_size
            ], dim=-1)
            q_attn = q_.transpose(0, 1).reshape(num_heads // world_size, seq_len, head_size).contiguous()
            k_attn = k_.transpose(0, 1).reshape(num_kv_heads // world_size, seq_len, head_size).contiguous()
            v_attn = v_.transpose(0, 1).reshape(num_kv_heads // world_size, seq_len, head_size).contiguous()
            k_attn = k_attn.repeat_interleave(num_heads // num_kv_heads, dim=0)
            v_attn = v_attn.repeat_interleave(num_heads // num_kv_heads, dim=0)
            
            # 2. Profile Core Attention computation
            start_event_attn.record()
            c_ = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            end_event_attn.record()
            
            c_pack = c_.view(num_heads // world_size * head_size, seq_len).transpose(0, 1).contiguous()
            c_out = torch.empty_like(c_pack)
            
            # 3. Profile All-to-All 2
            start_event_a2a2.record()
            dist.all_to_all_single(c_out, c_pack)
            end_event_a2a2.record()
            
            torch.cuda.synchronize()
            
            a2a1_times.append(start_event_a2a1.elapsed_time(end_event_a2a1))
            attn_times.append(start_event_attn.elapsed_time(end_event_attn))
            a2a2_times.append(start_event_a2a2.elapsed_time(end_event_a2a2))
            
        avg_a2a1 = sum(a2a1_times) / steps
        avg_attn = sum(attn_times) / steps
        avg_a2a2 = sum(a2a2_times) / steps
        
        total_comm = avg_a2a1 + avg_a2a2
        total_time = total_comm + avg_attn
        comm_pct = (total_comm / total_time) * 100
        
        if rank == 0:
            print(f"{seq_len:<10} | {avg_a2a1:<18.3f} | {avg_attn:<15.3f} | {avg_a2a2:<18.3f} | {comm_pct:<11.1f}%")

    if rank == 0:
        print("\n" + "=" * 80)
        print("          ASYNC COMMUNICATION-COMPUTATION OVERLAP PROFILING (C2)")
        print("=" * 80)
        print("Testing if async all_to_all_single can overlap with FFN/MLP GEMMs...")
        print("-" * 80)
        print(f"{'Seq Len':<10} | {'Non-Overlapped (ms)':<20} | {'Overlapped (ms)':<18} | {'Speedup':<10}")
        print("-" * 80)

    # Overlap Test:
    # We simulate FFN GEMMs (e.g. projecting tokens: [seq_len_per_gpu, hidden_size] -> [seq_len_per_gpu, FFN_intermediate_size])
    # while running async all-to-all.
    hidden_size = 4096
    ffn_size = 11008  # Llama-8B FFN intermediate size
    
    for seq_len in seq_lens:
        seq_len_per_gpu = seq_len // world_size
        
        # Data for all-to-all
        qkv = torch.randn((world_size, seq_len_per_gpu, total_head_dim // world_size), device=device, dtype=torch.float16)
        qkv_out = torch.empty_like(qkv)
        
        # Data for FFN GEMM (mock computation)
        x = torch.randn((seq_len_per_gpu, hidden_size), device=device, dtype=torch.float16)
        w1 = torch.randn((hidden_size, ffn_size), device=device, dtype=torch.float16)
        w2 = torch.randn((ffn_size, hidden_size), device=device, dtype=torch.float16)
        
        # CUDA Events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup FFN and communication
        for _ in range(5):
            # Sync mode
            dist.all_to_all_single(qkv_out, qkv)
            h1 = torch.matmul(x, w1)
            h2 = torch.matmul(h1, w2)
            
        # 1. Profile Non-Overlapped (Sync All-to-All + FFN Compute)
        non_overlap_times = []
        for _ in range(20):
            start_event.record()
            dist.all_to_all_single(qkv_out, qkv)
            h1 = torch.matmul(x, w1)
            h2 = torch.matmul(h1, w2)
            end_event.record()
            torch.cuda.synchronize()
            non_overlap_times.append(start_event.elapsed_time(end_event))
            
        # 2. Profile Overlapped (Async All-to-All + FFN Compute running concurrently)
        overlap_times = []
        for _ in range(20):
            start_event.record()
            # Start Async Communication
            handle = dist.all_to_all_single(qkv_out, qkv, async_op=True)
            # Run FFN computation concurrently on the GPU
            h1 = torch.matmul(x, w1)
            h2 = torch.matmul(h1, w2)
            # Wait for communication to finish
            handle.wait()
            end_event.record()
            torch.cuda.synchronize()
            overlap_times.append(start_event.elapsed_time(end_event))
            
        avg_non_overlap = sum(non_overlap_times) / 20
        avg_overlap = sum(overlap_times) / 20
        speedup = avg_non_overlap / avg_overlap
        
        if rank == 0:
            print(f"{seq_len:<10} | {avg_non_overlap:<20.3f} | {avg_overlap:<18.3f} | {speedup:<10.2f}x")
            
    if rank == 0:
        print("=" * 80)

if __name__ == "__main__":
    # Initialize the distributed environment
    # Expects torchrun configuration (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    dist.init_process_group("nccl")
    run_profiling()
    dist.destroy_process_group()
