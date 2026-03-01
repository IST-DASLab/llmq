import torch
from . import _pyllmq


_TORCH_TO_TYPE_NAME = {
    torch.float32:   "float32",
    torch.float16:   "float16",
    torch.bfloat16:  "bfloat16",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2:   "fp8_e5m2",
    torch.int32:     "int32",
    torch.int16:     "int16",
    torch.int8:      "int8",
}


# Encoder
@torch.library.custom_op("llmq::encoder_forward", mutates_args=("out",))
def encoder_forward(out: torch.Tensor, inp: torch.Tensor, wte: torch.Tensor, wpe: torch.Tensor | None = None, stream: int = 0) -> None:
    _pyllmq.encoder_forward(out, inp, wte, wpe, stream)


# RMSNorm

def get_rmsnorm_backward_scratch_size(channels: int):
    return _pyllmq.get_rmsnorm_backward_scratch_size(channels)

@torch.library.custom_op("llmq::rmsnorm_forward", mutates_args=("out", "rms", "absmax"))
def rmsnorm_forward(out: torch.Tensor, rms: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, absmax: torch.Tensor | None, epsilon: float, stream: int = 0) -> None:
    _pyllmq.rmsnorm_forward(out, rms, inp, weight, absmax, epsilon, stream)


@torch.library.custom_op("llmq::rmsnorm_backward", mutates_args=("dinp", "dweight", "scratch", "dresidual", "absmax"))
def rmsnorm_backward(dinp: torch.Tensor, dweight: torch.Tensor, scratch: torch.Tensor, dresidual: torch.Tensor,
                     dout: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor,
                     absmax: torch.Tensor | None, stream: int = 0) -> None:
    _pyllmq.rmsnorm_backward(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd, absmax, stream)


# Fused residual + RMSNorm
@torch.library.custom_op("llmq::fused_residual_rmsnorm_forward", mutates_args=("residual", "normed", "rrms", "absmax"))
def fused_residual_rmsnorm_forward(residual: torch.Tensor, normed: torch.Tensor, rrms: torch.Tensor,
                                   inp1: torch.Tensor, inp2: torch.Tensor, weight: torch.Tensor,
                                   absmax: torch.Tensor | None, epsilon: float, stream: int = 0) -> None:
    _pyllmq.fused_residual_rmsnorm_forward(residual, normed, rrms, inp1, inp2, weight, absmax, epsilon, stream)


# RoPE
@torch.library.custom_op("llmq::rope_forward", mutates_args=("out", "absmax"))
def rope_forward(out: torch.Tensor, x: torch.Tensor, freqs_cis: torch.Tensor, absmax: torch.Tensor | None,
                 Nq: int, Nkv: int, stream: int = 0) -> None:
    _pyllmq.rope_forward(out, x, freqs_cis, absmax, Nq, Nkv, stream)


@torch.library.custom_op("llmq::rope_backward", mutates_args=("dinp",  "absmax"))
def rope_backward(dinp: torch.Tensor, dout: torch.Tensor, freqs_cis: torch.Tensor, absmax: torch.Tensor | None,
                  Nq: int, Nkv: int, stream: int = 0) -> None:
    _pyllmq.rope_backward(dinp, dout, freqs_cis, absmax, Nq, Nkv, stream)


# SwiGLU
@torch.library.custom_op("llmq::swiglu_forward", mutates_args=("out", "absmax"))
def swiglu_forward(out: torch.Tensor, inp: torch.Tensor, absmax: torch.Tensor | None, stream: int = 0) -> None:
    _pyllmq.swiglu_forward(out, inp, absmax, stream)


@torch.library.custom_op("llmq::swiglu_forward_quant", mutates_args=("out", "scale"))
def swiglu_forward_quant(out: torch.Tensor, scale: torch.Tensor, inp: torch.Tensor, absmax: torch.Tensor | None, stream: int = 0) -> None:
    _pyllmq.swiglu_forward_quant(out, scale, inp, absmax, stream)


@torch.library.custom_op("llmq::swiglu_backward", mutates_args=("dinp", "absmax"))
def swiglu_backward(dinp: torch.Tensor, dout: torch.Tensor, inp: torch.Tensor, absmax: torch.Tensor | None, stream: int = 0) -> None:
    _pyllmq.swiglu_backward(dinp, dout, inp, absmax, stream)


# Attention (cuDNN)
@torch.library.custom_op("llmq::attention_forward", mutates_args=("out", "stats", "workspace"))
def attention_forward(out: torch.Tensor, stats: torch.Tensor, inp: torch.Tensor, workspace: torch.Tensor,
                      cudnn_handle: int, Hq: int, Hkv: int, stream: int = 0) -> None:
    _pyllmq.attention_forward(out, stats, inp, workspace, cudnn_handle, Hq, Hkv, stream)


@torch.library.custom_op("llmq::attention_backward", mutates_args=("dqkv",))
def attention_backward(dqkv: torch.Tensor, stats: torch.Tensor, out: torch.Tensor, dout: torch.Tensor, qkv: torch.Tensor,
                       workspace: torch.Tensor, cudnn_handle: int, Hq: int, Hkv: int, stream: int = 0) -> None:
    _pyllmq.attention_backward(dqkv, stats, out, dout, qkv, workspace, cudnn_handle, Hq, Hkv, stream)


# Classifier
@torch.library.custom_op("llmq::fused_classifier", mutates_args=("logits", "losses", "lse"))
def fused_classifier(logits: torch.Tensor, losses: torch.Tensor, lse: torch.Tensor, dloss: float,
                     targets: torch.Tensor, z_reg: float, write_dlogits: bool, stream: int = 0) -> None:
    _pyllmq.fused_classifier(logits, losses, lse, dloss, targets, z_reg, write_dlogits, stream)


@torch.library.custom_op("llmq::grouped_loss_sum", mutates_args=("out",))
def grouped_loss_sum(out: torch.Tensor, per_token_loss: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.grouped_loss_sum(out, per_token_loss, stream)


# Matmul and bias
@torch.library.custom_op("llmq::matmul", mutates_args=("c", "workspace"))
def matmul(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None,
           scale_a: torch.Tensor | None, scale_b: torch.Tensor | None,
           cublaslt_handle: int, workspace: torch.Tensor, mode: int,
           accumulate: bool = False, stream: int = 0) -> None:
    _pyllmq.matmul(c, a, b, bias, scale_a, scale_b, cublaslt_handle, workspace, mode, accumulate, stream)


def create_cublas_handle() -> int:
    return _pyllmq.create_cublas_handle()


@torch.library.custom_op("llmq::backward_bias", mutates_args=("dbias", "dbias_buffer"))
def backward_bias(dbias: torch.Tensor, dout: torch.Tensor, scale_a: torch.Tensor | None,
                  scale_b: torch.Tensor | None, dbias_buffer: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.backward_bias(dbias, dout, scale_a, scale_b, dbias_buffer, stream)


def get_bias_backward_scratch_size(dtype: torch.dtype, OC: int):
    return _pyllmq.get_bias_backward_scratch_size(_TORCH_TO_TYPE_NAME[dtype], OC)


# Quantization utils
@torch.library.custom_op("llmq::abs_max", mutates_args=("scale",))
def abs_max(scale: torch.Tensor, x: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.abs_max(scale, x, stream)


@torch.library.custom_op("llmq::quantize_with_abs_max", mutates_args=("out", "scale"))
def quantize_with_abs_max(out: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, abs_max: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.quantize_with_abs_max(out, scale, x, abs_max, stream)


@torch.library.custom_op("llmq::quantize_and_transpose_with_abs_max", mutates_args=("out", "scale"))
def quantize_and_transpose_with_abs_max(out: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, abs_max: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.quantize_and_transpose_with_abs_max(out, scale, x, abs_max, stream)


# Transpose
@torch.library.custom_op("llmq::transpose", mutates_args=("dst",))
def transpose(dst: torch.Tensor, src: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.transpose(dst, src, stream)


# Vector ops
@torch.library.custom_op("llmq::vector_add_sr", mutates_args=("dest",))
def vector_add_sr(dest: torch.Tensor, left: torch.Tensor, right: torch.Tensor, scale: float, seed: int, stream: int = 0) -> None:
    _pyllmq.vector_add_sr(dest, left, right, scale, seed, stream)


@torch.library.custom_op("llmq::vector_reduce_sr", mutates_args=("dest",))
def vector_reduce_sr(dest: torch.Tensor, src: torch.Tensor, scale: float, n_shards: int,
                     skip: int = -1, accumulate: bool = False, seed: int = 0, stream: int = 0) -> None:
    """
    Interprets `src` as a tensor of `n_shard` shards of size `nelem` each. The shards are summed together, and the result is either written to (`accumulate = false`)
    or added into (`accumulate = true`) `dest`, after being scaled by `scale`. All intermediate calculations are done in float precision, and stochastic rounding using the
    provided `seed` is applied before writing to `dest`. The `skip` parameter allows to skip one of the shards. Set to -1 to disable skipping.
    """
    _pyllmq.vector_reduce_sr(dest, src, scale, n_shards, skip, accumulate, seed, stream)


# Fillers
@torch.library.custom_op("llmq::fill_normal", mutates_args=("dst",))
def fill_normal(dst: torch.Tensor, mean: float, std: float, seed: int, subsequence: int = 0, stream: int = 0) -> None:
    _pyllmq.fill_normal(dst, mean, std, seed, subsequence, stream)


@torch.library.custom_op("llmq::fill_constant", mutates_args=("dst",))
def fill_constant(dst: torch.Tensor, value: float, stream: int = 0) -> None:
    _pyllmq.fill_constant(dst, value, stream)


# Encoder backward
@torch.library.custom_op("llmq::encoder_backward", mutates_args=("dwte", "scratch", "workload_indices", "bucket_info"))
def encoder_backward(dwte: torch.Tensor, scratch: torch.Tensor, workload_indices: torch.Tensor, bucket_info: torch.Tensor,
                     dout: torch.Tensor, inp: torch.Tensor, inputs_cpu: torch.Tensor, seed: int,
                     stream: int = 0, sync_event: int = 0, copy_stream: int = 0) -> None:
    _pyllmq.encoder_backward(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu, seed, stream, sync_event, copy_stream)


# Global norm
@torch.library.custom_op("llmq::global_norm_squared", mutates_args=("out",))
def global_norm_squared(out: torch.Tensor, values: torch.Tensor, stream: int = 0) -> None:
    _pyllmq.global_norm_squared(out, values, stream)


# AdamW update
# NOTE: can't have default values for mutable args, so we add a level of indirection here
@torch.library.custom_op("llmq::adamw_update", mutates_args=("params", "m", "v", "m_scales", "abs_max"))
def adamw_update_impl(params: torch.Tensor, grads: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                      learning_rate: float, beta1: float, beta2: float, t: int, eps: float, weight_decay: float,
                      grad_scale: torch.Tensor, m_scales: torch.Tensor | None,
                      abs_max: torch.Tensor | None, seed: int = 0, stream: int = 0) -> None:
    _pyllmq.adamw_update(params, grads, m, v, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, m_scales, abs_max, seed, stream)


def adamw_update(params: torch.Tensor, grads: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                 learning_rate: float, beta1: float, beta2: float, t: int, eps: float, weight_decay: float,
                 grad_scale: torch.Tensor, m_scales: torch.Tensor | None = None,
                 abs_max: torch.Tensor | None = None, seed: int = 0, stream: int = 0) -> None:
    adamw_update_impl(params, grads, m, v, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, m_scales, abs_max, seed, stream)


__all__ = [
    'encoder_forward',
    'rmsnorm_forward', 'rmsnorm_backward', 'fused_residual_rmsnorm_forward',
    'rope_forward', 'rope_backward',
    'swiglu_forward', 'swiglu_forward_quant', 'swiglu_backward',
    'attention_forward', 'attention_backward',
    'fused_classifier', 'grouped_loss_sum',
    'matmul', 'backward_bias',
    'abs_max', 'quantize_with_abs_max', 'quantize_and_transpose_with_abs_max',
    'transpose', 'vector_add_sr', 'vector_reduce_sr',
    'fill_normal', 'fill_constant',
    'encoder_backward', 'global_norm_squared', 'adamw_update',
]
