import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

import vllm
from vllm import LLM, SamplingParams

# Ensure general plugins are loaded
vllm.plugins.load_general_plugins()


# ============================================================
# Data Models
# ============================================================

@dataclass
class ExperimentConfig:
    model: str
    tp_size: int
    sp_size: int
    threshold: int
    workload: str
    tokens_count: int

    long_tokens: int
    short_tokens: int
    short_requests: int


@dataclass
class BenchmarkResult:
    workload_type: str
    prompt_count: int
    actual_prompt_tokens: int
    generated_tokens: int
    elapsed_time: float
    throughput: float


# ============================================================
# Workload Generators
# ============================================================

class WorkloadGenerator(ABC):

    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens

    @abstractmethod
    def generate(self):
        pass


class PrefillWorkload(WorkloadGenerator):
    """
    One long request.
    """

    def generate(self):
        prompt = (
            "Explain the concept of parallel and distributed computing in detail. "
            * (self.total_tokens // 10)
        )

        return [prompt]


class DecodeWorkload(WorkloadGenerator):
    """
    Many short requests.
    """

    def generate(self):
        return ["Hello world."] * self.total_tokens


class ImbalanceWorkload(WorkloadGenerator):
    """
    One long request + multiple short requests.

    Simulates token imbalance inside a batch.
    """

    def __init__(
        self,
        tokenizer,
        long_tokens,
        short_tokens,
        short_requests,
    ):
        self.tokenizer = tokenizer
        self.long_tokens = long_tokens
        self.short_tokens = short_tokens
        self.short_requests = short_requests

    def generate(self):

        def build_prompt_with_target_tokens(
            tokenizer,
            target_tokens: int,
        ) -> str:

            base_text = (
                "Explain the concept of parallel and distributed computing in detail. "
            )

            prompt = ""

            while len(tokenizer.encode(prompt)) < target_tokens:
                prompt += base_text

            return prompt

        long_prompt = build_prompt_with_target_tokens(
            self.tokenizer,
            self.long_tokens,
        )

        short_prompt = build_prompt_with_target_tokens(
            self.tokenizer,
            self.short_tokens,
        )

        prompts = [long_prompt]

        prompts.extend(
            [short_prompt]
            * self.short_requests
        )

        return prompts


def create_workload_generator(
    workload_type: str,
    total_tokens: int,
    **kwargs,
) -> WorkloadGenerator:

    generators = {
        "prefill": PrefillWorkload,
        "decode": DecodeWorkload,
        "imbalance": ImbalanceWorkload,
    }

    cls = generators[workload_type]

    if workload_type == "imbalance":
        return cls(**kwargs)

    return cls(total_tokens)


# ============================================================
# Metrics
# ============================================================

class MetricsCalculator:

    @staticmethod
    def calculate_throughput(
        total_prompt_tokens: int,
        generated_tokens: int,
        elapsed_time: float,
    ) -> float:

        if elapsed_time <= 0:
            return 0.0

        return (
            total_prompt_tokens + generated_tokens
        ) / elapsed_time


# ============================================================
# Utility
# ============================================================

def analyze_token_distribution(
    tokenizer,
    prompts,
):
    token_counts = [
        len(tokenizer.encode(prompt))
        for prompt in prompts
    ]

    return {
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": sum(token_counts) / len(token_counts),
        "ratio": max(token_counts) / max(1, min(token_counts)),
        "distribution": token_counts,
    }

def calculate_prompt_tokens(
    tokenizer,
    prompts,
) -> int:

    return sum(
        len(tokenizer.encode(prompt))
        for prompt in prompts
    )


def build_result(
    workload_type: str,
    prompt_count: int,
    actual_prompt_tokens: int,
    generated_tokens: int,
    elapsed_time: float,
) -> BenchmarkResult:

    throughput = MetricsCalculator.calculate_throughput(
        total_prompt_tokens=actual_prompt_tokens,
        generated_tokens=generated_tokens,
        elapsed_time=elapsed_time,
    )

    return BenchmarkResult(
        workload_type=workload_type,
        prompt_count=prompt_count,
        actual_prompt_tokens=actual_prompt_tokens,
        generated_tokens=generated_tokens,
        elapsed_time=elapsed_time,
        throughput=throughput,
    )


def print_result(result: BenchmarkResult, imbalance_analysis=None):

    if imbalance_analysis:
        print("\n" + "=" * 50)
        print("IMBALANCE WORKLOAD ANALYSIS")
        print("Prompt Token Distribution")
        print(
            imbalance_analysis["distribution"][:20] 
        )

        print(
            f"Min Tokens: {imbalance_analysis['min']}"
        )

        print(
            f"Max Tokens: {imbalance_analysis['max']}"  
        )

        print(
            f"Mean Tokens: {imbalance_analysis['mean']:.2f}" 
        )

        print(
            f"Imbalance Ratio: {imbalance_analysis['ratio']:.2f}x"  
        )

    print("\n" + "=" * 50)
    print("EXPERIMENTAL METRICS & RESULTS")
    print("=" * 50)

    print(f"Workload Type:           {result.workload_type.upper()}")
    print(f"Prompt Count:            {result.prompt_count}")
    print(f"Actual Prompt Tokens:    {result.actual_prompt_tokens}")
    print(f"Generated Tokens:        {result.generated_tokens}")
    print(f"Execution Time:          {result.elapsed_time:.4f} seconds")
    print(f"Throughput:              {result.throughput:.2f} tokens/s")

    print("=" * 50)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="PDP Final Project: Token Imbalance & Shift Parallelism Benchmark"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor Parallel size",
    )

    parser.add_argument(
        "--sp-size",
        type=int,
        default=2,
        help="Ulysses Sequence Parallel size",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=1024,
        help=(
            "Shift Parallel threshold. "
            "0 = force SP, large value = force TP"
        ),
    )

    parser.add_argument(
        "--workload",
        type=str,
        choices=[
            "prefill",
            "decode",
            "imbalance",
        ],
        required=True,
    )

    parser.add_argument(
        "--tokens-count",
        type=int,
        default=800,
        help="Target workload size",
    )

    parser.add_argument(
        "--long-tokens",
        type=int,
        default=8192,
        help="Target token count for the long request",
    )

    parser.add_argument(
        "--short-tokens",
        type=int,
        default=8,
        help="Target token count for each short request",
    )

    parser.add_argument(
        "--short-requests",
        type=int,
        default=100,
        help="Number of short requests",
    )

    return parser.parse_args()


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(config: ExperimentConfig):

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    print(
        f"Initializing LLM engine "
        f"(TP={config.tp_size}, "
        f"SP={config.sp_size}, "
        f"Threshold={config.threshold})..."
    )

    llm = LLM(
        model=config.model,
        tensor_parallel_size=config.tp_size,
        ulysses_sequence_parallel_size=config.sp_size,
        enable_shift_parallel=True,
        shift_parallel_threshold=config.threshold,
        max_model_len=100000,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": True,
        },
    )

    tokenizer = llm.get_tokenizer()

    generator = create_workload_generator(
        config.workload,
        config.tokens_count,
        tokenizer=tokenizer,
        long_tokens=config.long_tokens,
        short_tokens=config.short_tokens,
        short_requests=config.short_requests,
    )

    prompts = generator.generate()

    actual_prompt_tokens = calculate_prompt_tokens(
        tokenizer,
        prompts,
    )

    imbalance_analysis = (
        analyze_token_distribution(tokenizer, prompts)
        if config.workload == "imbalance"
        else None
    )

    # Warmup
    print("Running warmup...")
    llm.generate(
        ["Warmup prompt"],
        sampling_params,
    )

    print(
        f"Running benchmark workload "
        f"'{config.workload}' "
        f"with {len(prompts)} request(s)..."
    )

    start_time = time.perf_counter()

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    total_generated = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
    )

    result = build_result(
        workload_type=config.workload,
        prompt_count=len(prompts),
        actual_prompt_tokens=actual_prompt_tokens,
        generated_tokens=total_generated,
        elapsed_time=elapsed_time,
    )

    print_result(result, imbalance_analysis)

    # Cleanup
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    except Exception:
        pass

    del llm

    time.sleep(2)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    config = ExperimentConfig(
        model=args.model,
        tp_size=args.tp_size,
        sp_size=args.sp_size,
        threshold=args.threshold,
        workload=args.workload,
        tokens_count=args.tokens_count,
        long_tokens=args.long_tokens,
        short_tokens=args.short_tokens,
        short_requests=args.short_requests,
    )

    run_benchmark(config)