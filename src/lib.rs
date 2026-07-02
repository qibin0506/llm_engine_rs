use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub use_qk_norm: bool,
    pub attention_qkv_bias: bool,
    pub attention_out_bias: bool,
    pub mlp_bias: bool,
    pub lm_head_bias: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

fn get_linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb).map_err(E::msg)
    } else {
        linear_no_bias(in_dim, out_dim, vb).map_err(E::msg)
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x.dtype())?.broadcast_mul(&self.weight)
    }
}

fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let half_dim = dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

    let neg_x2 = x2.neg()?;
    let rotated_x = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;

    let x_cos = x.broadcast_mul(cos)?;
    let rot_sin = rotated_x.broadcast_mul(sin)?;

    Ok((x_cos + rot_sin)?)
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    theta: f32,
    pub kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let q_norm = if cfg.use_qk_norm { Some(RmsNorm::load(cfg.head_dim, cfg.norm_eps, vb.pp("q_norm"))?) } else { None };
        let k_norm = if cfg.use_qk_norm { Some(RmsNorm::load(cfg.head_dim, cfg.norm_eps, vb.pp("k_norm"))?) } else { None };

        Ok(Self {
            q_proj: get_linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, cfg.attention_qkv_bias, vb.pp("q_proj"))?,
            k_proj: get_linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, cfg.attention_qkv_bias, vb.pp("k_proj"))?,
            v_proj: get_linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, cfg.attention_qkv_bias, vb.pp("v_proj"))?,
            o_proj: get_linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, cfg.attention_out_bias, vb.pp("o_proj"))?,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rot_dim: (cfg.head_dim as f32 * cfg.partial_rotary_factor) as usize,
            theta: cfg.rope_theta,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let device = x.device();

        let mut q = self.q_proj.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let mut k = self.k_proj.forward(x)?.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = self.v_proj.forward(x)?.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        if let Some(norm) = &self.q_norm { q = norm.forward(&q)?; }
        if let Some(norm) = &self.k_norm { k = norm.forward(&k)?; }

        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;

        let mut inv_freqs = Vec::new();
        for i in (0..self.rot_dim).step_by(2) {
            inv_freqs.push(1.0 / self.theta.powf(i as f32 / self.rot_dim as f32));
        }
        let mut freqs_vec = Vec::new();
        for p in pos..(pos + seq_len) {
            for &inv_f in &inv_freqs { freqs_vec.push((p as f32) * inv_f); }
        }
        let freqs = Tensor::from_vec(freqs_vec, (seq_len, self.rot_dim / 2), device)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.reshape((1, 1, seq_len, self.rot_dim))?;
        let sin = emb.sin()?.reshape((1, 1, seq_len, self.rot_dim))?;

        let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
        let q_rot_embed = apply_rotary_pos_emb(&q_rot, &cos, &sin)?;
        let q = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?;

        let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
        let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
        let k_rot_embed = apply_rotary_pos_emb(&k_rot, &cos, &sin)?;
        let k = Tensor::cat(&[&k_rot_embed, &k_pass], D::Minus1)?;

        // KV Cache 存储更新
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let new_k = Tensor::cat(&[prev_k, &k], 2)?;
                let new_v = Tensor::cat(&[prev_v, &v], 2)?;
                (new_k, new_v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // ===============================================
        // 关键修复：GQA KV Repeat_Interleave
        // 通过 unsqueeze -> repeat -> reshape 完美对齐 PyTorch
        // ===============================================
        let k = if self.num_heads != self.num_kv_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let (b, kv_heads, s_len, d) = k.dims4()?;
            k.unsqueeze(2)? // [b, kv_heads, 1, s_len, d]
             .repeat((1, 1, repeat, 1, 1))? // [b, kv_heads, repeat, s_len, d]
             .reshape((b, self.num_heads, s_len, d))?
        } else { k };

        let v = if self.num_heads != self.num_kv_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let (b, kv_heads, s_len, d) = v.dims4()?;
            v.unsqueeze(2)?
             .repeat((1, 1, repeat, 1, 1))?
             .reshape((b, self.num_heads, s_len, d))?
        } else { v };
        // ===============================================

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // 修复版 Causal Mask (兼容多次 chunked prefill)
        let kv_seq_len = k.dim(2)?;
        let attn_weights = if seq_len > 1 {
            let mask: Vec<_> = (0..seq_len).flat_map(|i| {
                (0..kv_seq_len).map(move |j| {
                    if j > i + pos { f32::NEG_INFINITY } else { 0f32 }
                })
            }).collect();
            let mask = Tensor::from_vec(mask, (seq_len, kv_seq_len), device)?
                       .reshape((1, 1, seq_len, kv_seq_len))?
                       .to_dtype(attn_weights.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: get_linear(cfg.hidden_size, cfg.intermediate_size, cfg.mlp_bias, vb.pp("gate_proj"))?,
            up_proj: get_linear(cfg.hidden_size, cfg.intermediate_size, cfg.mlp_bias, vb.pp("up_proj"))?,
            down_proj: get_linear(cfg.intermediate_size, cfg.hidden_size, cfg.mlp_bias, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = (gate.clone() * candle_nn::ops::sigmoid(&gate)?)?;
        self.down_proj.forward(&(act * up)?)
    }
}

struct DecoderLayer {
    pub attn: Attention,
    mlp: Mlp,
    attn_norm: RmsNorm,
    mlp_norm: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(cfg, vb.pp("attn"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            attn_norm: RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("attn_norm"))?,
            mlp_norm: RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.attn_norm.forward(x)?;
        let x = self.attn.forward(&x, pos)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.mlp_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((x + residual)?)
    }
}

struct LlmModel {
    embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    head_norm: RmsNorm,
    lm_head: Linear,
}

impl LlmModel {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(cfg, vb.pp(&format!("layers.{}", i)))?);
        }
        let head_norm = RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("head_norm"))?;

        let lm_head = if vb.contains_tensor("lm_head.weight") {
            get_linear(cfg.hidden_size, cfg.vocab_size, cfg.lm_head_bias, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self { embed_tokens, layers, head_norm, lm_head })
    }

    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, pos)?;
        }
        let x = self.head_norm.forward(&x)?;
        let x = x.i((.., x.dim(1)? - 1, ..))?;
        Ok(self.lm_head.forward(&x)?)
    }
}

// ==========================================
// 采样与生成 API
// ==========================================

fn sample_logits(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut StdRng,
) -> u32 {
    if temperature > 0.0 && (temperature - 1.0).abs() > 1e-4 {
        for l in logits.iter_mut() {
            *l /= temperature;
        }
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

    if top_k > 0 && top_k < probs.len() {
        for &idx in indices.iter().skip(top_k) {
            probs[idx] = 0.0;
        }
    }

    if top_p > 0.0 && top_p < 1.0 {
        let mut cumsum = 0.0;
        let mut cutoff = indices.len();
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum > top_p {
                cutoff = i + 1;
                break;
            }
        }
        for &idx in indices.iter().skip(cutoff) {
            probs[idx] = 0.0;
        }
    }

    if let Ok(dist) = WeightedIndex::new(&probs) {
        dist.sample(rng) as u32
    } else {
        indices[0] as u32
    }
}

#[pyclass]
pub struct LlmEngine {
    model: LlmModel,
    device: Device,
}

#[pymethods]
impl LlmEngine {
    #[new]
    pub fn new(weights_path: &str, config_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let cfg = Config {
            vocab_size: config_dict.get_item("vocab_size")?.expect("Missing vocab_size").extract()?,
            hidden_size: config_dict.get_item("hidden_size")?.expect("Missing hidden_size").extract()?,
            intermediate_size: config_dict.get_item("intermediate_size")?.expect("Missing intermediate_size").extract()?,
            num_hidden_layers: config_dict.get_item("num_hidden_layers")?.expect("Missing num_hidden_layers").extract()?,
            num_attention_heads: config_dict.get_item("num_attention_heads")?.expect("Missing num_attention_heads").extract()?,
            num_key_value_heads: config_dict.get_item("num_key_value_heads")?.map(|v| v.extract().unwrap()).unwrap_or_else(|| config_dict.get_item("num_attention_heads").unwrap().unwrap().extract().unwrap()),
            head_dim: config_dict.get_item("head_dim")?.map(|v| v.extract().unwrap()).unwrap_or_else(|| {
                let h: usize = config_dict.get_item("hidden_size").unwrap().unwrap().extract().unwrap();
                let n: usize = config_dict.get_item("num_attention_heads").unwrap().unwrap().extract().unwrap();
                h / n
            }),
            norm_eps: config_dict.get_item("norm_eps")?.map(|v| v.extract().unwrap_or(1e-6)).unwrap_or(1e-6),
            use_qk_norm: config_dict.get_item("use_qk_norm")?.map(|v| v.extract().unwrap_or(true)).unwrap_or(true),
            attention_qkv_bias: config_dict.get_item("attention_qkv_bias")?.map(|v| v.extract().unwrap_or(false)).unwrap_or(false),
            attention_out_bias: config_dict.get_item("attention_out_bias")?.map(|v| v.extract().unwrap_or(false)).unwrap_or(false),
            mlp_bias: config_dict.get_item("mlp_bias")?.map(|v| v.extract().unwrap_or(false)).unwrap_or(false),
            lm_head_bias: config_dict.get_item("lm_head_bias")?.map(|v| v.extract().unwrap_or(false)).unwrap_or(false),
            rope_theta: config_dict.get_item("rope_theta")?.map(|v| v.extract().unwrap_or(10000.0)).unwrap_or(10000.0),
            partial_rotary_factor: config_dict.get_item("partial_rotary_factor")?.map(|v| v.extract().unwrap_or(1.0)).unwrap_or(1.0),
        };

        let device = Device::Cpu; 
        let dtype = DType::F32;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))? };
        let model = LlmModel::load(&cfg, vb).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { model, device })
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.model.layers {
            layer.attn.kv_cache = None;
        }
    }

    // 保留原有的完整生成 API
    #[pyo3(signature = (input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, exclude_penalty_tokens=None, suppress_tokens=None, eos_token_id=2, seed=42))]
    pub fn generate(
        &mut self,
        input_ids: Vec<u32>,
        max_new_tokens: usize,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
        exclude_penalty_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
        eos_token_id: u32,
        seed: u64,
    ) -> PyResult<Vec<u32>> {
        // (省略内部代码，和你之前贴的一模一样)
        self.clear_kv_cache();
        let temp = temperature.unwrap_or(1.0);
        let top_k = top_k.unwrap_or(0);
        let top_p = top_p.unwrap_or(1.0);
        let rep_penalty = repetition_penalty.unwrap_or(1.0);
        let exclude_tokens: HashSet<u32> = exclude_penalty_tokens.unwrap_or_default().into_iter().collect();
        let suppress_set: HashSet<u32> = suppress_tokens.unwrap_or_default().into_iter().collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_pos = 0;
        let mut current_input = input_ids.clone();

        for _ in 0..max_new_tokens {
            let input_tensor = Tensor::new(current_input.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let logits_tensor = self.model.forward(&input_tensor, current_pos).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let mut logits = logits_tensor.squeeze(0).and_then(|t| t.to_vec1::<f32>()).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            for &token in &suppress_set {
                if (token as usize) < logits.len() { logits[token as usize] = f32::NEG_INFINITY; }
            }

            if rep_penalty > 0.0 && (rep_penalty - 1.0).abs() > 1e-4 {
                for &token in &generated_tokens {
                    if !exclude_tokens.contains(&token) && (token as usize) < logits.len() {
                        let score = logits[token as usize];
                        logits[token as usize] = if score < 0.0 { score * rep_penalty } else { score / rep_penalty };
                    }
                }
            }

            let next_token = if temp > 0.0 { sample_logits(&mut logits, temp, top_k, top_p, &mut rng) } else { logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32 };
            generated_tokens.push(next_token);
            if next_token == eos_token_id { break; }
            current_pos += current_input.len();
            current_input = vec![next_token]; 
        }
        Ok(generated_tokens)
    }

    /// =========================================
    /// ✨ 新增：流式生成 API，返回一个 Python 迭代器
    /// =========================================
    #[pyo3(signature = (input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, exclude_penalty_tokens=None, suppress_tokens=None, eos_token_id=2, seed=42))]
    pub fn streaming_generate(
        slf: Py<Self>,
        py: Python,
        input_ids: Vec<u32>,
        max_new_tokens: usize,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
        exclude_penalty_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
        eos_token_id: u32,
        seed: u64,
    ) -> PyResult<LlmStreamer> {
        // 先借用底层引擎清空缓存
        slf.borrow_mut(py).clear_kv_cache();

        let exclude_tokens: HashSet<u32> = exclude_penalty_tokens.unwrap_or_default().into_iter().collect();
        let suppress_tokens: HashSet<u32> = suppress_tokens.unwrap_or_default().into_iter().collect();

        // 构造返回给 Python 的状态机
        Ok(LlmStreamer {
            engine: slf.clone(),
            rng: StdRng::seed_from_u64(seed),
            current_input: input_ids,
            current_pos: 0,
            generated_tokens: Vec::new(),
            tokens_generated_count: 0,
            max_new_tokens,
            temperature: temperature.unwrap_or(1.0),
            top_k: top_k.unwrap_or(0),
            top_p: top_p.unwrap_or(1.0),
            repetition_penalty: repetition_penalty.unwrap_or(1.0),
            exclude_tokens,
            suppress_tokens,
            eos_token_id,
        })
    }
}

/// 这是一个实现了 Python Iterator 协议的 Rust 结构体
#[pyclass(unsendable)]
pub struct LlmStreamer {
    engine: Py<LlmEngine>,
    rng: StdRng,
    current_input: Vec<u32>,
    current_pos: usize,
    generated_tokens: Vec<u32>,
    tokens_generated_count: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    exclude_tokens: HashSet<u32>,
    suppress_tokens: HashSet<u32>,
    eos_token_id: u32,
}

#[pymethods]
impl LlmStreamer {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python) -> PyResult<Option<u32>> {
        // 达到生成上限，终止迭代 (抛出 StopIteration)
        if slf.tokens_generated_count >= slf.max_new_tokens {
            return Ok(None);
        }

        // 获取底层 LlmEngine 的可变引用
        let mut engine = slf.engine.borrow_mut(py);

        // --- 1. 前向传播 ---
        let input_tensor = Tensor::new(slf.current_input.as_slice(), &engine.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let logits_tensor = engine.model.forward(&input_tensor, slf.current_pos)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut logits = logits_tensor.squeeze(0)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // --- 2. 应用各种惩罚 ---
        for &token in &slf.suppress_tokens {
            if (token as usize) < logits.len() {
                logits[token as usize] = f32::NEG_INFINITY;
            }
        }

        if slf.repetition_penalty > 0.0 && (slf.repetition_penalty - 1.0).abs() > 1e-4 {
            for &token in &slf.generated_tokens {
                if !slf.exclude_tokens.contains(&token) && (token as usize) < logits.len() {
                    let score = logits[token as usize];
                    logits[token as usize] = if score < 0.0 { score * slf.repetition_penalty } else { score / slf.repetition_penalty };
                }
            }
        }

        // --- 3. 采样 ---
        let next_token = if slf.temperature > 0.0 {
            sample_logits(&mut logits, slf.temperature, slf.top_k, slf.top_p, &mut slf.rng)
        } else {
            logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
        };

        // --- 4. 维护状态机 ---
        slf.generated_tokens.push(next_token);
        slf.tokens_generated_count += 1;
        
        slf.current_pos += slf.current_input.len();
        slf.current_input = vec![next_token];

        if next_token == slf.eos_token_id {
            // 标记下次迭代直接结束
            slf.tokens_generated_count = slf.max_new_tokens; 
        }

        // 将当前 token yield 给 Python
        Ok(Some(next_token))
    }
}

#[pymodule]
fn llm_engine_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LlmEngine>()?;
    m.add_class::<LlmStreamer>()?; // 注册流式发生器
    Ok(())
}
