use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: Option<usize>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub n_dense_layer: Option<usize>,
}

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
    pub max_position_embeddings: usize,
    pub moe_config: Option<MoEConfig>,

    pub attention_type: String,
    pub hybrid_ratio: String,
    pub hybrid_softmax_head_dim: Option<usize>,
    pub hybrid_softmax_num_heads: Option<usize>,
    pub hybrid_softmax_num_kv_heads: Option<usize>,
    pub hybrid_softmax_gated: bool,
    pub gated_deltanet_use_short_conv: bool,
    pub gated_deltanet_conv_kernel_size: usize,
    pub gated_deltanet_conv_bias: bool,
}

#[derive(PartialEq)]
enum LayerType {
    Softmax,
    GatedDeltaNet,
}

fn resolve_layer_type(cfg: &Config, layer_idx: usize) -> LayerType {
    if cfg.attention_type != "hybrid" {
        return LayerType::Softmax;
    }
    let parts: Vec<&str> = cfg.hybrid_ratio.split(':').collect();
    let gd_count = parts.get(0).and_then(|s| s.parse::<usize>().ok()).unwrap_or(3);
    let sa_count = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let cycle_len = gd_count + sa_count;
    if cycle_len == 0 {
        return LayerType::Softmax;
    }
    if layer_idx % cycle_len < gd_count {
        LayerType::GatedDeltaNet
    } else {
        LayerType::Softmax
    }
}

#[derive(Clone, Default)]
pub struct LayerCache {
    pub kv: Option<(Tensor, Tensor)>,
    pub recurrent_state: Option<Tensor>,
    pub conv_q: Option<Tensor>,
    pub conv_k: Option<Tensor>,
    pub conv_v: Option<Tensor>,
}

#[derive(Clone, Default)]
pub struct KVCache {
    pub layers: Vec<LayerCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerCache::default(); num_layers],
        }
    }
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            *l = LayerCache::default();
        }
    }
}

fn get_linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb).map_err(E::msg)
    } else {
        linear_no_bias(in_dim, out_dim, vb).map_err(E::msg)
    }
}

fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let threshold = x.maximum(&zeros)?;
    let exp_neg_abs = x.abs()?.neg()?.exp()?;
    (threshold + (exp_neg_abs + 1.0)?.log()?).map_err(E::msg)
}

fn l2_norm(x: &Tensor) -> Result<Tensor> {
    let sqr_sum = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_div(&(sqr_sum + 1e-6)?.sqrt()?).map_err(E::msg)
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

struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let rope_head_dim = if cfg.attention_type == "hybrid" {
            cfg.hybrid_softmax_head_dim.unwrap_or(cfg.head_dim)
        } else {
            cfg.head_dim
        };
        let dim = (rope_head_dim as f32 * cfg.partial_rotary_factor) as usize;
        let theta = cfg.rope_theta;
        let mut inv_freqs = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            inv_freqs.push(1.0 / theta.powf(i as f32 / dim as f32));
        }
        let inv_freq = Tensor::from_vec(inv_freqs, dim / 2, device)?;
        Ok(Self { inv_freq })
    }

    fn forward(&self, pos: usize, seq_len: usize, dtype: DType) -> Result<(Tensor, Tensor)> {
        let device = self.inv_freq.device();
        let pos_ids = Tensor::arange(pos as u32, (pos + seq_len) as u32, device)?.to_dtype(dtype)?;
        let freqs = pos_ids.unsqueeze(1)?.matmul(&self.inv_freq.to_dtype(dtype)?.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.reshape((1, 1, seq_len, ()))?;
        let sin = emb.sin()?.reshape((1, 1, seq_len, ()))?;
        Ok((cos, sin))
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

struct CausalMasker {
    mask: Tensor,
}

impl CausalMasker {
    fn new(max_seq_len: usize, device: &Device, dtype: DType) -> Result<Self> {
        let mut mask_vec = vec![0.0f32; max_seq_len * max_seq_len];
        for i in 0..max_seq_len {
            for j in 0..max_seq_len {
                if j > i {
                    mask_vec[i * max_seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_vec, (max_seq_len, max_seq_len), device)?.to_dtype(dtype)?;
        Ok(Self { mask })
    }

    fn get_mask(&self, seq_len: usize, kv_seq_len: usize) -> Result<Tensor> {
        let pos = kv_seq_len - seq_len;
        Ok(self.mask.narrow(0, pos, seq_len)?.narrow(1, 0, kv_seq_len)?
            .reshape((1, 1, seq_len, kv_seq_len))?)
    }
}

struct ShortConv {
    conv: candle_nn::Conv1d,
    kernel_size: usize,
}

impl ShortConv {
    fn load(d: usize, k: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: d };
        let conv = if bias {
            candle_nn::conv1d(d, d, k, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(d, d, k, cfg, vb)?
        };
        Ok(Self { conv, kernel_size: k })
    }

    fn forward(&self, x: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        let x_t = x.transpose(1, 2)?;
        let (out, new_state) = if let Some(prev) = state {
            let x_cat = Tensor::cat(&[prev as &Tensor, &x_t], D::Minus1)?;
            let out = self.conv.forward(&x_cat)?;
            let new_state = x_cat.narrow(D::Minus1, x_cat.dim(D::Minus1)? - (self.kernel_size - 1), self.kernel_size - 1)?;
            (out, new_state)
        } else {
            let (b, c, _) = x_t.dims3()?;
            let zeros = Tensor::zeros((b, c, self.kernel_size - 1), x_t.dtype(), x_t.device())?;
            let x_padded = Tensor::cat(&[&zeros, &x_t], D::Minus1)?;
            let out = self.conv.forward(&x_padded)?;
            let new_state = x_padded.narrow(D::Minus1, x_padded.dim(D::Minus1)? - (self.kernel_size - 1), self.kernel_size - 1)?;
            (out, new_state)
        };
        *state = Some(new_state);
        out.transpose(1, 2).map_err(E::msg)
    }
}

struct GatedDeltaNet {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    beta_proj: Linear,
    alpha_proj: Linear,
    g_proj: Linear,
    o_proj: Linear,
    g_norm: RmsNorm,
    dt_bias: Tensor,
    a_log: Tensor,
    conv_q: Option<ShortConv>,
    conv_k: Option<ShortConv>,
    conv_v: Option<ShortConv>,
    num_heads: usize,
    head_dim: usize,
}

impl GatedDeltaNet {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;

        let conv_q = if cfg.gated_deltanet_use_short_conv { Some(ShortConv::load(num_heads * head_dim, cfg.gated_deltanet_conv_kernel_size, cfg.gated_deltanet_conv_bias, vb.pp("conv_q.conv"))?) } else { None };
        let conv_k = if cfg.gated_deltanet_use_short_conv { Some(ShortConv::load(num_heads * head_dim, cfg.gated_deltanet_conv_kernel_size, cfg.gated_deltanet_conv_bias, vb.pp("conv_k.conv"))?) } else { None };
        let conv_v = if cfg.gated_deltanet_use_short_conv { Some(ShortConv::load(num_heads * head_dim, cfg.gated_deltanet_conv_kernel_size, cfg.gated_deltanet_conv_bias, vb.pp("conv_v.conv"))?) } else { None };

        Ok(Self {
            q_proj: get_linear(cfg.hidden_size, num_heads * head_dim, false, vb.pp("q_proj"))?,
            k_proj: get_linear(cfg.hidden_size, num_heads * head_dim, false, vb.pp("k_proj"))?,
            v_proj: get_linear(cfg.hidden_size, num_heads * head_dim, false, vb.pp("v_proj"))?,
            beta_proj: get_linear(cfg.hidden_size, num_heads, false, vb.pp("beta_proj"))?,
            alpha_proj: get_linear(cfg.hidden_size, num_heads, false, vb.pp("alpha_proj"))?,
            g_proj: get_linear(cfg.hidden_size, num_heads * head_dim, false, vb.pp("g_proj"))?,
            o_proj: get_linear(num_heads * head_dim, cfg.hidden_size, false, vb.pp("o_proj"))?,
            g_norm: RmsNorm::load(head_dim, cfg.norm_eps, vb.pp("g_norm"))?,
            dt_bias: vb.get(num_heads, "dt_bias")?,
            a_log: vb.get(num_heads, "A_log")?,
            conv_q,
            conv_k,
            conv_v,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;
        let g = self.g_proj.forward(x)?;

        let beta = candle_nn::ops::sigmoid(&self.beta_proj.forward(x)?)?;
        let alpha_logits = self.alpha_proj.forward(x)?;

        if let Some(cq) = &self.conv_q { q = cq.forward(&q, &mut cache.conv_q)?; }
        if let Some(ck) = &self.conv_k { k = ck.forward(&k, &mut cache.conv_k)?; }
        if let Some(cv) = &self.conv_v { v = cv.forward(&v, &mut cache.conv_v)?; }

        let q = candle_nn::ops::silu(&q)?.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = candle_nn::ops::silu(&k)?.reshape((b, t, self.num_heads, self.head_dim))?;
        let v = candle_nn::ops::silu(&v)?.reshape((b, t, self.num_heads, self.head_dim))?;
        let g = g.reshape((b, t, self.num_heads, self.head_dim))?;

        let dt_b = self.dt_bias.reshape((1, 1, self.num_heads))?.broadcast_as((b, t, self.num_heads))?;
        let dt = stable_softplus(&(alpha_logits + dt_b)?)?;
        let a_log_exp = self.a_log.exp()?.reshape((1, 1, self.num_heads))?.broadcast_as((b, t, self.num_heads))?;
        let g_decay = (dt.neg()? * a_log_exp)?;
        let alpha = g_decay.exp()?;

        let q = l2_norm(&q)?;
        let k = l2_norm(&k)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q = (q * scale)?;

        let mut curr_state = match &cache.recurrent_state {
            Some(state) => state.clone(),
            None => Tensor::zeros((b, self.num_heads, self.head_dim, self.head_dim), q.dtype(), q.device())?
        };

        let mut out_list = Vec::with_capacity(t);
        for i in 0..t {
            let q_i = q.narrow(1, i, 1)?.squeeze(1)?;
            let k_i = k.narrow(1, i, 1)?.squeeze(1)?;
            let v_i = v.narrow(1, i, 1)?.squeeze(1)?;
            let a_i = alpha.narrow(1, i, 1)?.squeeze(1)?.reshape((b, self.num_heads, 1, 1))?;
            let b_i = beta.narrow(1, i, 1)?.squeeze(1)?.reshape((b, self.num_heads, 1))?;

            curr_state = curr_state.broadcast_mul(&a_i)?;

            let v_pred = k_i.unsqueeze(2)?.matmul(&curr_state)?.squeeze(2)?;
            let v_err = (v_i - v_pred)?;

            let b_v_err = v_err.broadcast_mul(&b_i)?;
            let delta_s = k_i.unsqueeze(3)?.matmul(&b_v_err.unsqueeze(2)?)?;
            curr_state = (curr_state + delta_s)?;

            let o_i = q_i.unsqueeze(2)?.matmul(&curr_state)?.squeeze(2)?;
            out_list.push(o_i);
        }

        cache.recurrent_state = Some(curr_state);

        let out = Tensor::stack(&out_list, 1)?;
        let out = self.g_norm.forward(&out)?;
        let out = (out * candle_nn::ops::silu(&g)?)?;
        let out = out.reshape((b, t, self.num_heads * self.head_dim))?;

        Ok(self.o_proj.forward(&out)?)
    }
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
    is_gated: bool,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let is_hybrid_softmax = cfg.attention_type == "hybrid" && resolve_layer_type(cfg, layer_idx) == LayerType::Softmax;

        let num_heads = if is_hybrid_softmax { cfg.hybrid_softmax_num_heads.unwrap_or(cfg.num_attention_heads) } else { cfg.num_attention_heads };
        let num_kv_heads = if is_hybrid_softmax { cfg.hybrid_softmax_num_kv_heads.unwrap_or(cfg.num_key_value_heads) } else { cfg.num_key_value_heads };
        let head_dim = if is_hybrid_softmax { cfg.hybrid_softmax_head_dim.unwrap_or(cfg.head_dim) } else { cfg.head_dim };
        let is_gated = is_hybrid_softmax && cfg.hybrid_softmax_gated;

        let q_norm = if cfg.use_qk_norm { Some(RmsNorm::load(head_dim, cfg.norm_eps, vb.pp("q_norm"))?) } else { None };
        let k_norm = if cfg.use_qk_norm { Some(RmsNorm::load(head_dim, cfg.norm_eps, vb.pp("k_norm"))?) } else { None };

        let q_proj_out = if is_gated { num_heads * head_dim * 2 } else { num_heads * head_dim };

        Ok(Self {
            q_proj: get_linear(cfg.hidden_size, q_proj_out, cfg.attention_qkv_bias, vb.pp("q_proj"))?,
            k_proj: get_linear(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_qkv_bias, vb.pp("k_proj"))?,
            v_proj: get_linear(cfg.hidden_size, num_kv_heads * head_dim, cfg.attention_qkv_bias, vb.pp("v_proj"))?,
            o_proj: get_linear(num_heads * head_dim, cfg.hidden_size, cfg.attention_out_bias, vb.pp("o_proj"))?,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rot_dim: (head_dim as f32 * cfg.partial_rotary_factor) as usize,
            is_gated,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, cache: &mut LayerCache) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_out = self.q_proj.forward(x)?;
        let (mut q, gate) = if self.is_gated {
            let q_out = q_out.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_out.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;
            (q, Some(gate))
        } else {
            (q_out.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?, None)
        };

        let mut k = self.k_proj.forward(x)?.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let mut v = self.v_proj.forward(x)?.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        if let Some(norm) = &self.q_norm { q = norm.forward(&q)?; }
        if let Some(norm) = &self.k_norm { k = norm.forward(&k)?; }

        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;
        v = v.transpose(1, 2)?;

        let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
        let q_rot_embed = apply_rotary_pos_emb(&q_rot, cos, sin)?;
        let q = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?;

        let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
        let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
        let k_rot_embed = apply_rotary_pos_emb(&k_rot, cos, sin)?;
        let k = Tensor::cat(&[&k_rot_embed, &k_pass], D::Minus1)?;

        let (k, v) = match &cache.kv {
            Some((prev_k, prev_v)) => {
                let new_k = Tensor::cat(&[prev_k as &Tensor, &k], 2)?;
                let new_v = Tensor::cat(&[prev_v as &Tensor, &v], 2)?;
                (new_k, new_v)
            }
            None => (k, v),
        };
        cache.kv = Some((k.clone(), v.clone()));

        let k = if self.num_heads != self.num_kv_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let (b, _kv_heads, s_len, d) = k.dims4()?;
            k.unsqueeze(2)?.repeat((1, 1, repeat, 1, 1))?.reshape((b, self.num_heads, s_len, d))?
        } else { k };

        let v = if self.num_heads != self.num_kv_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let (b, _kv_heads, s_len, d) = v.dims4()?;
            v.unsqueeze(2)?.repeat((1, 1, repeat, 1, 1))?.reshape((b, self.num_heads, s_len, d))?
        } else { v };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            attn_weights = attn_weights.broadcast_add(m)?;
        }

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let mut attn_output = attn_weights.matmul(&v)?;

        attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        if let Some(g) = gate {
            attn_output = (attn_output * candle_nn::ops::sigmoid(&g.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?))?;
        }

        Ok(self.o_proj.forward(&attn_output)?)
    }
}

enum TokenMixer {
    Softmax(Attention),
    GatedDeltaNet(GatedDeltaNet),
}

impl TokenMixer {
    fn load(cfg: &Config, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        match resolve_layer_type(cfg, layer_idx) {
            LayerType::Softmax => Ok(Self::Softmax(Attention::load(cfg, vb, layer_idx)?)),
            LayerType::GatedDeltaNet => Ok(Self::GatedDeltaNet(GatedDeltaNet::load(cfg, vb)?)),
        }
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, cache: &mut LayerCache) -> Result<Tensor> {
        match self {
            Self::Softmax(attn) => attn.forward(x, cos, sin, mask, cache),
            Self::GatedDeltaNet(gdn) => gdn.forward(x, cache), // GatedDeltaNet does not need rope/mask for standard recurrence
        }
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Config, vb: VarBuilder, intermediate_size: Option<usize>) -> Result<Self> {
        let int_size = intermediate_size.unwrap_or(cfg.intermediate_size);
        Ok(Self {
            gate_proj: get_linear(cfg.hidden_size, int_size, cfg.mlp_bias, vb.pp("gate_proj"))?,
            up_proj: get_linear(cfg.hidden_size, int_size, cfg.mlp_bias, vb.pp("up_proj"))?,
            down_proj: get_linear(int_size, cfg.hidden_size, cfg.mlp_bias, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        self.down_proj.forward(&(silu_gate * up)?)
    }
}

fn topk_2d(t: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let (b, _) = t.dims2()?;
    let t_f32 = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut topk_vals = Vec::with_capacity(b * k);
    let mut topk_ids = Vec::with_capacity(b * k);

    for row in t_f32.iter() {
        let mut indexed: Vec<(usize, &f32)> = row.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..k {
            topk_vals.push(*indexed[i].1);
            topk_ids.push(indexed[i].0 as u32);
        }
    }
    let vals = Tensor::from_vec(topk_vals, (b, k), t.device())?.to_dtype(t.dtype())?;
    let ids = Tensor::from_vec(topk_ids, (b, k), t.device())?;
    Ok((vals, ids))
}

struct MoE {
    experts: Vec<Mlp>,
    shared_experts: Option<Mlp>,
    gate_weight: Tensor,
    top_k: usize,
    n_routed_experts: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl MoE {
    fn load(cfg: &Config, moe_cfg: &MoEConfig, vb: VarBuilder) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..moe_cfg.n_routed_experts {
            experts.push(Mlp::load(cfg, vb.pp(format!("experts.{}", i)), Some(moe_cfg.intermediate_size))?);
        }

        let shared_experts = if let Some(n_shared) = moe_cfg.n_shared_experts {
            let shared_size = moe_cfg.intermediate_size * n_shared;
            Some(Mlp::load(cfg, vb.pp("shared_experts"), Some(shared_size))?)
        } else {
            None
        };

        let gate_weight = vb.get((moe_cfg.n_routed_experts, cfg.hidden_size), "gate.weight")?;

        Ok(Self {
            experts,
            shared_experts,
            gate_weight,
            top_k: moe_cfg.num_experts_per_tok,
            n_routed_experts: moe_cfg.n_routed_experts,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            norm_topk_prob: moe_cfg.norm_topk_prob,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;

        let logits = x_flat.to_dtype(DType::F32)?.matmul(&self.gate_weight.t()?.to_dtype(DType::F32)?)?;
        let scores = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let (mut topk_weights, topk_ids) = topk_2d(&scores, self.top_k)?;

        if self.norm_topk_prob && self.top_k > 1 {
            let sum = topk_weights.sum_keepdim(D::Minus1)?;
            topk_weights = topk_weights.broadcast_div(&sum)?;
        }
        topk_weights = (topk_weights * self.routed_scaling_factor as f64)?;

        let mut out_tokens = Vec::with_capacity(b * s);
        let x_split = x_flat.chunk(b * s, 0)?;
        let ids_split = topk_ids.chunk(b * s, 0)?;
        let weights_split = topk_weights.chunk(b * s, 0)?;

        for (i, token) in x_split.iter().enumerate() {
            let ids = ids_split[i].to_vec1::<u32>()?;
            let weights = weights_split[i].to_vec1::<f32>()?;

            let mut combined_out = token.zeros_like()?;
            for j in 0..self.top_k {
                let exp_idx = ids[j] as usize;
                let exp_out = self.experts[exp_idx].forward(token)?;
                combined_out = (combined_out + (exp_out * weights[j] as f64)?)?;
            }
            out_tokens.push(combined_out);
        }

        let mut out = Tensor::cat(&out_tokens, 0)?;

        if let Some(shared) = &self.shared_experts {
            out = (out + shared.forward(&x_flat)?)?;
        }

        Ok(out.reshape((b, s, h))?)
    }
}

enum FeedForward {
    Mlp(Mlp),
    MoE(MoE),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => Ok(mlp.forward(x)?),
            Self::MoE(moe) => moe.forward(x),
        }
    }
}

struct DecoderLayer {
    pub mixer: TokenMixer,
    mlp: FeedForward,
    attn_norm: RmsNorm,
    mlp_norm: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &Config, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let use_moe = if let Some(moe_cfg) = &cfg.moe_config {
            moe_cfg.n_dense_layer.map_or(true, |n| layer_idx >= n)
        } else {
            false
        };

        let mlp = if use_moe {
            FeedForward::MoE(MoE::load(cfg, cfg.moe_config.as_ref().unwrap(), vb.pp("mlp"))?)
        } else {
            FeedForward::Mlp(Mlp::load(cfg, vb.pp("mlp"), None)?)
        };

        Ok(Self {
            mixer: TokenMixer::load(cfg, vb.pp("attn"), layer_idx)?,
            mlp,
            attn_norm: RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("attn_norm"))?,
            mlp_norm: RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, _pos: usize, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, cache: &mut LayerCache) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.attn_norm.forward(x)?;
        let x = self.mixer.forward(&x, cos, sin, mask, cache)?;
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
    rotary_emb: RotaryEmbedding,
    causal_masker: CausalMasker,
}

impl LlmModel {
    fn load(cfg: &Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(cfg, vb.pp(&format!("layers.{}", i)), i)?);
        }
        let head_norm = RmsNorm::load(cfg.hidden_size, cfg.norm_eps, vb.pp("head_norm"))?;

        let lm_head = if vb.contains_tensor("lm_head.weight") {
            get_linear(cfg.hidden_size, cfg.vocab_size, cfg.lm_head_bias, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        let rotary_emb = RotaryEmbedding::new(cfg, device)?;
        let causal_masker = CausalMasker::new(cfg.max_position_embeddings, device, dtype)?;

        Ok(Self { embed_tokens, layers, head_norm, lm_head, rotary_emb, causal_masker })
    }

    fn forward(&self, input_ids: &Tensor, pos: usize, kv_cache: &mut KVCache) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let seq_len = x.dim(1)?;
        let kv_seq_len = pos + seq_len;

        let (cos, sin) = self.rotary_emb.forward(pos, seq_len, x.dtype())?;

        let mask = if seq_len > 1 {
            Some(self.causal_masker.get_mask(seq_len, kv_seq_len)?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, pos, &cos, &sin, mask.as_ref(), &mut kv_cache.layers[i])?;
        }
        let x = self.head_norm.forward(&x)?;
        let x = x.i((.., x.dim(1)? - 1, ..))?;
        Ok(self.lm_head.forward(&x)?)
    }
}

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
    kv_cache: KVCache,
}

fn get_field<'py, T>(obj: &Bound<'py, PyAny>, key: &str) -> Option<T>
where
    for<'a> T: FromPyObject<'a, 'py>,
{
    if let Ok(item) = obj.get_item(key) {
        if !item.is_none() {
            if let Ok(val) = item.extract::<T>() {
                return Some(val);
            }
        }
    }
    if let Ok(attr) = obj.getattr(key) {
        if !attr.is_none() {
            if let Ok(val) = attr.extract::<T>() {
                return Some(val);
            }
        }
    }
    None
}

#[pymethods]
impl LlmEngine {
    #[new]
    #[pyo3(signature = (weights_path, config_dict, dtype="fp32", device="cpu"))]
    pub fn new(weights_path: &str, config_dict: &Bound<'_, PyDict>, dtype: &str, device: &str) -> PyResult<Self> {

        let moe_config = if let Ok(Some(moe)) = config_dict.get_item("moe_config") {
            if !moe.is_none() {
                let intermediate_size = get_field::<usize>(&moe, "intermediate_size");
                let n_routed_experts = get_field::<usize>(&moe, "n_routed_experts");
                let num_experts_per_tok = get_field::<usize>(&moe, "num_experts_per_tok");

                if let (Some(intermediate_size), Some(n_routed_experts), Some(num_experts_per_tok)) = (intermediate_size, n_routed_experts, num_experts_per_tok) {
                    let n_shared_experts = get_field::<usize>(&moe, "n_shared_experts");
                    let routed_scaling_factor = get_field::<f32>(&moe, "routed_scaling_factor").unwrap_or(1.0);
                    let norm_topk_prob = get_field::<bool>(&moe, "norm_topk_prob").unwrap_or(false);
                    let n_dense_layer = get_field::<usize>(&moe, "n_dense_layer");

                    Some(MoEConfig {
                        intermediate_size,
                        n_routed_experts,
                        n_shared_experts,
                        num_experts_per_tok,
                        routed_scaling_factor,
                        norm_topk_prob,
                        n_dense_layer,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let vocab_size: usize = config_dict.get_item("vocab_size")?.expect("Missing vocab_size").extract()?;
        let hidden_size: usize = config_dict.get_item("hidden_size")?.expect("Missing hidden_size").extract()?;
        let intermediate_size: usize = config_dict.get_item("intermediate_size")?.expect("Missing intermediate_size").extract()?;
        let num_hidden_layers: usize = config_dict.get_item("num_hidden_layers")?.expect("Missing num_hidden_layers").extract()?;
        let num_attention_heads: usize = config_dict.get_item("num_attention_heads")?.expect("Missing num_attention_heads").extract()?;

        let num_key_value_heads: usize = config_dict.get_item("num_key_value_heads")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(num_attention_heads);

        let head_dim: usize = config_dict.get_item("head_dim")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let rope_obj = config_dict.get_item("rope_config")?.filter(|v| !v.is_none());
        let rope_theta: f32 = rope_obj.as_ref()
            .and_then(|r| get_field::<f32>(r, "rope_theta"))
            .or_else(|| config_dict.get_item("rope_theta").ok().flatten().and_then(|v| v.extract().ok()))
            .unwrap_or(10000.0);

        let partial_rotary_factor: f32 = rope_obj.as_ref()
            .and_then(|r| get_field::<f32>(r, "partial_rotary_factor"))
            .or_else(|| config_dict.get_item("partial_rotary_factor").ok().flatten().and_then(|v| v.extract().ok()))
            .unwrap_or(1.0);

        let gd_obj = config_dict.get_item("gated_deltanet_config")?.filter(|v| !v.is_none());
        let gated_deltanet_use_short_conv: bool = gd_obj.as_ref()
            .and_then(|v| get_field::<bool>(v, "use_short_conv"))
            .unwrap_or(true);
        let gated_deltanet_conv_kernel_size: usize = gd_obj.as_ref()
            .and_then(|v| get_field::<usize>(v, "conv_kernel_size"))
            .unwrap_or(4);
        let gated_deltanet_conv_bias: bool = gd_obj.as_ref()
            .and_then(|v| get_field::<bool>(v, "conv_bias"))
            .unwrap_or(false);

        let cfg = Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            norm_eps: config_dict.get_item("norm_eps")?.and_then(|v| v.extract().ok()).unwrap_or(1e-6),
            use_qk_norm: config_dict.get_item("use_qk_norm")?.and_then(|v| v.extract().ok()).unwrap_or(true),
            attention_qkv_bias: config_dict.get_item("attention_qkv_bias")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            attention_out_bias: config_dict.get_item("attention_out_bias")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            mlp_bias: config_dict.get_item("mlp_bias")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            lm_head_bias: config_dict.get_item("lm_head_bias")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            rope_theta,
            partial_rotary_factor,
            max_position_embeddings: config_dict.get_item("max_position_embeddings")?.and_then(|v| v.extract().ok()).unwrap_or(4096),
            moe_config,

            attention_type: config_dict.get_item("attention_type")?.and_then(|v| v.extract().ok()).unwrap_or_else(|| "softmax".to_string()),
            hybrid_ratio: config_dict.get_item("hybrid_ratio")?.and_then(|v| v.extract().ok()).unwrap_or_else(|| "3:1".to_string()),
            hybrid_softmax_head_dim: config_dict.get_item("hybrid_softmax_head_dim")?.and_then(|v| v.extract().ok()),
            hybrid_softmax_num_heads: config_dict.get_item("hybrid_softmax_num_heads")?.and_then(|v| v.extract().ok()),
            hybrid_softmax_num_kv_heads: config_dict.get_item("hybrid_softmax_num_kv_heads")?.and_then(|v| v.extract().ok()),
            hybrid_softmax_gated: config_dict.get_item("hybrid_softmax_gated")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            gated_deltanet_use_short_conv,
            gated_deltanet_conv_kernel_size,
            gated_deltanet_conv_bias,
        };

        let dev_str = device.to_lowercase();
        let device = if dev_str.starts_with("cuda") {
            let idx = dev_str.split(':').nth(1).unwrap_or("0").parse::<usize>().unwrap_or(0);
            Device::new_cuda(idx).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CUDA initialization failed: {}", e)))?
        } else if dev_str == "metal" || dev_str == "mps" {
            Device::new_metal(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Metal/MPS initialization failed: {}", e)))?
        } else {
            Device::Cpu
        };

        let dt = match dtype.to_lowercase().as_str() {
            "fp16" | "float16" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            _ => DType::F32,
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dt, &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))? };
        let model = LlmModel::load(&cfg, vb, &device, dt).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let kv_cache = KVCache::new(cfg.num_hidden_layers);

        Ok(Self { model, device, kv_cache })
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.clear();
    }

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
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let logits_tensor = self.model.forward(&input_tensor, current_pos, &mut self.kv_cache)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut logits = logits_tensor
                .to_dtype(DType::F32).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .squeeze(0)
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            for &token in &suppress_set {
                if (token as usize) < logits.len() {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
            }

            if rep_penalty > 0.0 && (rep_penalty - 1.0).abs() > 1e-4 {
                for &token in &generated_tokens {
                    if !exclude_tokens.contains(&token) && (token as usize) < logits.len() {
                        let score = logits[token as usize];
                        logits[token as usize] = if score < 0.0 { score * rep_penalty } else { score / rep_penalty };
                    }
                }
            }

            let next_token = if temp > 0.0 {
                sample_logits(&mut logits, temp, top_k, top_p, &mut rng)
            } else {
                logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
            };

            generated_tokens.push(next_token);

            if next_token == eos_token_id {
                break;
            }

            current_pos += current_input.len();
            current_input = vec![next_token];
        }

        Ok(generated_tokens)
    }

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
        slf.borrow_mut(py).clear_kv_cache();

        let exclude_tokens: HashSet<u32> = exclude_penalty_tokens.unwrap_or_default().into_iter().collect();
        let suppress_tokens: HashSet<u32> = suppress_tokens.unwrap_or_default().into_iter().collect();

        Ok(LlmStreamer {
            engine: slf.clone_ref(py),
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
        if slf.tokens_generated_count >= slf.max_new_tokens {
            return Ok(None);
        }

        let mut logits = {
            let mut engine_py = slf.engine.borrow_mut(py);
            let engine = &mut *engine_py;

            let input_tensor = Tensor::new(slf.current_input.as_slice(), &engine.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let logits_tensor = engine.model.forward(&input_tensor, slf.current_pos, &mut engine.kv_cache)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            logits_tensor
                .to_dtype(DType::F32).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .squeeze(0)
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };


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

        let next_token = if slf.temperature > 0.0 {
            sample_logits(&mut logits, slf.temperature, slf.top_k, slf.top_p, &mut slf.rng)
        } else {
            logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
        };

        slf.generated_tokens.push(next_token);
        slf.tokens_generated_count += 1;

        slf.current_pos += slf.current_input.len();
        slf.current_input = vec![next_token];

        if next_token == slf.eos_token_id {
            slf.tokens_generated_count = slf.max_new_tokens;
        }

        Ok(Some(next_token))
    }
}

#[pymodule]
fn llm_engine_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LlmEngine>()?;
    m.add_class::<LlmStreamer>()?;
    Ok(())
}