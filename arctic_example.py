import vllm
from vllm import LLM, SamplingParams

vllm.plugins.load_general_plugins()

llm = LLM(
    model="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
#    quantization="float16",
    tensor_parallel_size=2,
    ulysses_sequence_parallel_size=2,
    enable_shift_parallel=True,
    max_model_len=1024,
    gpu_memory_utilization=0.70,
    enforce_eager=True,    
    speculative_config={
      "method": "suffix",
      "num_speculative_tokens": 3,
      "enable_suffix_decoding": True,
    },
)

conversation = [
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=800)

outputs = llm.chat(conversation, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)

del llm