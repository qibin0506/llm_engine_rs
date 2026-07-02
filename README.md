# llm_engine_rs

``` python
config_dict = {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "norm_eps": 1e-6,  # 对应 config.norm_eps 的默认值
    "use_qk_norm": True,
    "attention_qkv_bias": False,  # 默认 False
    "attention_out_bias": False,  # 默认 False
    "mlp_bias": False,  # 默认 False
    "lm_head_bias": False,  # 默认 False
    "rope_theta": 1000000.0,  # 注意这里是 100万
    "partial_rotary_factor": 1.0  # 默认 1.0
}

# 3. 极速加载 Rust 推理引擎
print("🚀 正在通过 Rust 引擎加载模型...")
engine = llm_engine_rs.LlmEngine("sft.safetensors", config_dict)
print("✅ 模型加载成功！")

prompt = get_eval_prompt('你好，请你用kotlin写一个Hello World。')
output_ids = engine.generate(
    input_ids=TrainerTools().tokenizer.encode(prompt),
    max_new_tokens=2048,
    temperature=0.85,
    top_k=30,
    top_p=0.85,
    # repetition_penalty=1.05,
    eos_token_id=TrainerTools().tokenizer.end
)

output_text = TrainerTools().tokenizer.decode(output_ids)
print(output_text)
```
