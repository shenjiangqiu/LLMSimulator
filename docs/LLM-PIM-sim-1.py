from dataclasses import dataclass
from math import log2, ceil
from typing import Dict, List


# ============================================================
# Near-bank PIM device
# ============================================================

@dataclass
class NearBankPIMDevice:
    n_stacks: int
    banks_per_stack: int
    active_banks_per_stack: int | None = None

    # local bank-side bandwidth per active bank PE
    internal_bank_bw_GBs: float = 16.0

    # PE microarchitecture
    macs_per_pe: int = 16
    pe_frequency_GHz: float = 1.0
    ops_per_mac_per_cycle: int = 2   # 1 MAC = 2 FLOPs by default

    # efficiencies
    bw_eff: float = 0.85
    flop_eff: float = 0.80

    # precision scaling for compute throughput (16-bit is base = 1x)
    throughput_scale_int8: float = 2.0
    throughput_scale_int4: float = 4.0
    throughput_scale_int2: float = 8.0

    # system movement / reduction parameters
    hbm_stack_host_bw_GBs: float = 819.2
    reduce_bw_GBs: float = 512.0
    reduce_latency_ns: float = 15.0
    sync_latency_ns: float = 40.0
    handoff_link_GBs: float | None = None
    
    # Q*K^T dot-product operand op counts.
    # For a BF16-by-x-bit dot term, the Q operand contributes
    # k_quant_op_scale_16 ops and the K operand contributes
    # k_quant_op_scale_x ops.
    k_quant_op_scale_16: float = 5.0
    k_quant_op_scale_x: float = 0.0
    v_quant_op_scale: float = 1.0
    
    def __post_init__(self) -> None:
        if self.active_banks_per_stack is None:
            self.active_banks_per_stack = self.banks_per_stack
        if self.handoff_link_GBs is None:
            self.handoff_link_GBs = self.total_host_bw_GBs()

        if not (1 <= self.active_banks_per_stack <= self.banks_per_stack):
            raise ValueError("active_banks_per_stack must be within [1, banks_per_stack].")

    def total_banks(self) -> int:
        return self.n_stacks * self.banks_per_stack

    def total_active_banks(self) -> int:
        return self.n_stacks * self.active_banks_per_stack

    def flops_per_pe_GFs(self) -> float:
        return self.macs_per_pe * self.ops_per_mac_per_cycle * self.pe_frequency_GHz

    def total_host_bw_GBs(self) -> float:
        return self.n_stacks * self.hbm_stack_host_bw_GBs

    def local_bw_GBs(self) -> float:
        return self.total_active_banks() * self.internal_bank_bw_GBs * self.bw_eff

    def local_flops_GFs(self) -> float:
        return self.total_active_banks() * self.flops_per_pe_GFs() * self.flop_eff

    def throughput_scale_for_kv_bits(self, kv_bits: int) -> float:
        if kv_bits <= 2:
            return self.throughput_scale_int2
        if kv_bits <= 4:
            return self.throughput_scale_int4
        if kv_bits <= 8:
            return self.throughput_scale_int8
        return 1.0

    def local_flops_GFs_for_kv_bits(self, kv_bits: int) -> float:
        return self.local_flops_GFs() 

    def peak_local_bw_GBs(self) -> float:
        return self.total_active_banks() * self.internal_bank_bw_GBs

    def peak_local_flops_GFs(self) -> float:
        return self.total_active_banks() * self.flops_per_pe_GFs()

    def roofline_time_s(self, bytes_moved: float, flops_fp16: float, flops_lowbit: float = 0.0, kv_bits: int = 16) -> float:
        """
        Compute roofline time for mixed-precision operations on PIM.
        Args:
            bytes_moved: total bytes moved
            flops_fp16: fp16 operations
            flops_lowbit: low-bit operations (0 if not used)
            kv_bits: bits for low-bit operations
        """
        t_mem = bytes_moved / (self.local_bw_GBs() * 1e9)
        
        if flops_lowbit > 0:
            # Mixed precision: compute time for each and take max (pipelined)
            t_cmp_fp16 = flops_fp16 / (self.local_flops_GFs() * 1e9)
            lowbit_throughput_scale = self.throughput_scale_for_kv_bits(kv_bits)
            t_cmp_lowbit = flops_lowbit / (self.local_flops_GFs() * lowbit_throughput_scale * 1e9)
            t_cmp = max(t_cmp_fp16, t_cmp_lowbit)
        else:
            t_cmp = flops_fp16 / (self.local_flops_GFs_for_kv_bits(kv_bits) * 1e9)
        
        return max(t_mem, t_cmp)

    def roofline_components_s(self, bytes_moved: float, flops_fp16: float, flops_lowbit: float = 0.0, kv_bits: int = 16) -> Dict[str, float]:
        """
        Compute roofline components for mixed-precision operations on PIM.
        Args:
            bytes_moved: total bytes moved
            flops_fp16: fp16 operations
            flops_lowbit: low-bit operations (0 if not used)
            kv_bits: bits for low-bit operations
        """
        t_mem = bytes_moved / (self.local_bw_GBs() * 1e9)
        
        if flops_lowbit > 0:
            # Mixed precision: compute time for each and track both
            t_cmp_fp16 = flops_fp16 / (self.local_flops_GFs() * 1e9)
            lowbit_throughput_scale = self.throughput_scale_for_kv_bits(kv_bits)
            t_cmp_lowbit = flops_lowbit / (self.local_flops_GFs() * lowbit_throughput_scale * 1e9)
            t_cmp = max(t_cmp_fp16, t_cmp_lowbit)
        else:
            t_cmp_fp16 = flops_fp16 / (self.local_flops_GFs_for_kv_bits(kv_bits) * 1e9)
            t_cmp_lowbit = 0.0
            t_cmp = t_cmp_fp16
        
        return {
            "t_mem_s": t_mem,
            "t_cmp_s": t_cmp,
            "t_cmp_fp16_s": t_cmp_fp16,
            "t_cmp_lowbit_s": t_cmp_lowbit,
            "t_exec_s": max(t_mem, t_cmp),
        }

    def broadcast_time_s(self, bytes_per_sequence: float, batch_size: int, use_host_link: bool = False) -> float:
        if use_host_link:
            return 0  # no broadcast time when using host link, as data is injected directly to each stack's HBM interface
        else:
            dist_bw = self.local_bw_GBs()
            fanout = 1
            return (fanout * bytes_per_sequence * batch_size) / (dist_bw * 1e9)
        
        
    def reduction_time_s(self, result_bytes: float) -> float:
        n = self.total_active_banks()
        if n <= 1:
            return 0.0
        stages = ceil(log2(n))
        t_bw = result_bytes / (self.reduce_bw_GBs * 1e9)
        t_lat = stages * self.reduce_latency_ns * 1e-9
        return t_bw + t_lat

    def sync_time_s(self) -> float:
        return self.sync_latency_ns * 1e-9

    def handoff_time_s(self, bytes_out: float) -> float:
        return bytes_out / (self.handoff_link_GBs * 1e9)

    def summary(self) -> Dict[str, float]:
        return {
            "n_stacks": self.n_stacks,
            "banks_per_stack": self.banks_per_stack,
            "active_banks_per_stack": self.active_banks_per_stack,
            "total_active_banks": self.total_active_banks(),
            "internal_bank_bw_GBs": self.internal_bank_bw_GBs,
            "pe_GFLOPs": self.flops_per_pe_GFs(),
            "overall_memory_bw_peak_GBs": self.peak_local_bw_GBs(),
            "overall_memory_bw_effective_GBs": self.local_bw_GBs(),
            "overall_compute_peak_GFs": self.peak_local_flops_GFs(),
            "overall_compute_effective_GFs": self.local_flops_GFs(),
            "overall_compute_effective_GFs_kv16": self.local_flops_GFs_for_kv_bits(16),
            "overall_compute_effective_GFs_kv8": self.local_flops_GFs_for_kv_bits(8),
            "overall_compute_effective_GFs_kv4": self.local_flops_GFs_for_kv_bits(4),
            "overall_compute_effective_GFs_kv2": self.local_flops_GFs_for_kv_bits(2),
            "host_hbm_aggregate_bw_GBs": self.total_host_bw_GBs(),
            "effective_local_bw_GBs": self.local_bw_GBs(),
            "effective_local_flops_GFs": self.local_flops_GFs(),
            "k_quant_op_scale_16": self.k_quant_op_scale_16,
            "k_quant_op_scale_x": self.k_quant_op_scale_x,
            "v_quant_op_scale": self.v_quant_op_scale,
        }


# ============================================================
# A100 GPU model
# ============================================================

@dataclass
class A100GPU:
    """
    A100 80GB PCIe style model.
    """
    model_name: str = "A100 80GB (PCIe)"
    memory_type: str = "HBM2e"
    mem_bw_GBs: float = 1935.0
    tensor_flops_GFs: float = 312_000.0
    bf16_tensor_tflops_dense: float = 312.0
    bf16_tensor_tflops_sparse: float = 624.0
    fp16_tensor_tflops_dense: float = 312.0
    fp16_tensor_tflops_sparse: float = 624.0
    int8_tensor_tops_dense: float = 624.0
    int8_tensor_tops_sparse: float = 1248.0
    int4_tensor_tops_dense: float = 1248.0
    int4_tensor_tops_sparse: float = 2496.0
    mem_eff: float = 0.85
    compute_eff: float = 0.35
    kernel_launch_us: float = 3.0

    def effective_mem_bw_GBs(self) -> float:
        return self.mem_bw_GBs * self.mem_eff

    def effective_flops_GFs(self) -> float:
        return self.tensor_flops_GFs * self.compute_eff

    def _get_fp16_throughput_GFs(self) -> float:
        """Return fp16 tensor throughput in GFLOP/s"""
        return self.fp16_tensor_tflops_dense * 1000.0 * self.compute_eff

    def _get_lowbit_throughput_GFs(self, kv_bits: int) -> float:
        """Return low-bit throughput in GTOPS/s based on kv_bits"""
        if kv_bits <= 4:
            return self.int4_tensor_tops_dense * 1000.0 * self.compute_eff
        elif kv_bits <= 8:
            return self.int8_tensor_tops_dense * 1000.0 * self.compute_eff
        else:
            # Fall back to fp16 if kv_bits > 8
            return self._get_fp16_throughput_GFs()

    def roofline_time_s(self, bytes_moved: float, flops_fp16: float, flops_lowbit: float = 0.0, kv_bits: int = 16, launches: int = 1) -> float:
        """
        Compute roofline time for mixed-precision operations.
        Args:
            bytes_moved: total bytes moved
            flops_fp16: fp16 operations
            flops_lowbit: low-bit operations (0 if not used)
            kv_bits: bits for low-bit operations (ignored if flops_lowbit == 0)
            launches: number of kernel launches
        """
        t_mem = bytes_moved / (self.effective_mem_bw_GBs() * 1e9)
        
        # Compute time for each precision
        if flops_lowbit > 0:
            t_cmp_fp16 = flops_fp16 / (self._get_fp16_throughput_GFs() * 1e9)
            t_cmp_lowbit = flops_lowbit / (self._get_lowbit_throughput_GFs(kv_bits) * 1e9)
            # Mixed precision: take max (pipelined/overlapped execution)
            t_cmp = max(t_cmp_fp16, t_cmp_lowbit)
        else:
            t_cmp = flops_fp16 / (self.effective_flops_GFs() * 1e9)
        
        t_launch = launches * self.kernel_launch_us * 1e-6
        return max(t_mem, t_cmp) + t_launch

    def roofline_components_s(self, bytes_moved: float, flops_fp16: float, flops_lowbit: float = 0.0, kv_bits: int = 16, launches: int = 1) -> Dict[str, float]:
        """
        Compute roofline components for mixed-precision operations.
        Args:
            bytes_moved: total bytes moved
            flops_fp16: fp16 operations
            flops_lowbit: low-bit operations (0 if not used)
            kv_bits: bits for low-bit operations (ignored if flops_lowbit == 0)
            launches: number of kernel launches
        """
        t_mem = bytes_moved / (self.effective_mem_bw_GBs() * 1e9)
        
        # Compute time for each precision
        if flops_lowbit > 0:
            t_cmp_fp16 = flops_fp16 / (self._get_fp16_throughput_GFs() * 1e9)
            t_cmp_lowbit = flops_lowbit / (self._get_lowbit_throughput_GFs(kv_bits) * 1e9)
            # Mixed precision: track both components
            t_cmp = max(t_cmp_fp16, t_cmp_lowbit)
        else:
            t_cmp_fp16 = flops_fp16 / (self.effective_flops_GFs() * 1e9)
            t_cmp_lowbit = 0.0
            t_cmp = t_cmp_fp16
        
        t_launch = launches * self.kernel_launch_us * 1e-6
        return {
            "t_mem_s": t_mem,
            "t_cmp_s": t_cmp,
            "t_cmp_fp16_s": t_cmp_fp16,
            "t_cmp_lowbit_s": t_cmp_lowbit,
            "t_launch_s": t_launch,
            "t_exec_s": max(t_mem, t_cmp) + t_launch,
        }

    def summary(self) -> Dict[str, float]:
        return {
            "model_name": self.model_name,
            "memory_type": self.memory_type,
            "mem_bw_GBs": self.mem_bw_GBs,
            "tensor_flops_GFs": self.tensor_flops_GFs,
            "bf16_tensor_tflops_dense": self.bf16_tensor_tflops_dense,
            "bf16_tensor_tflops_sparse": self.bf16_tensor_tflops_sparse,
            "fp16_tensor_tflops_dense": self.fp16_tensor_tflops_dense,
            "fp16_tensor_tflops_sparse": self.fp16_tensor_tflops_sparse,
            "int8_tensor_tops_dense": self.int8_tensor_tops_dense,
            "int8_tensor_tops_sparse": self.int8_tensor_tops_sparse,
            "int4_tensor_tops_dense": self.int4_tensor_tops_dense,
            "int4_tensor_tops_sparse": self.int4_tensor_tops_sparse,
            "effective_mem_bw_GBs": self.effective_mem_bw_GBs(),
            "effective_flops_GFs": self.effective_flops_GFs(),
        }


# ============================================================
# LLM config
# ============================================================

@dataclass
class LLMConfig:
    name: str
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_mult: float = 3.5
    bytes_act: int = 2
    kv_bits: int = 16
    kv_group_size: int = 64
    kv_meta_bytes_per_group: int = 4
    kv_quant_factor: float = 4.0 #4.0
    kv_dequant_opt_policy: str = "cooperative" # "ideal", "none-opt", "none-quant", "cooperative", ""


    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def d_ff(self) -> int:
        return int(self.ffn_mult * self.hidden_size)

    @property
    def kv_bytes(self) -> float:
        return self.kv_bits / 8.0


@dataclass
class DecodeRequest:
    batch_size: int
    context_len: int
    gen_len: int


# ============================================================
# Cost model
# ============================================================

class LLMDecodeCostModel:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def layer_proj_flops(self, B: int) -> float:
        d = self.cfg.d_model
        d_kv = self.cfg.d_kv
        return (
            2 * B * d * d +      # q_proj
            2 * B * d * d_kv +   # k_proj
            2 * B * d * d_kv +   # v_proj
            2 * B * d * d        # o_proj
        )

    def layer_proj_bytes(self, B: int) -> float:
        d = self.cfg.d_model
        d_kv = self.cfg.d_kv
        b = self.cfg.bytes_act
        return B * (d + d + d_kv + d_kv + d) * b

    def layer_mlp_flops(self, B: int) -> float:
        d = self.cfg.d_model
        dff = self.cfg.d_ff
        return (
            2 * B * d * dff +
            2 * B * d * dff +
            2 * B * dff * d
        )

    def layer_mlp_bytes(self, B: int) -> float:
        d = self.cfg.d_model
        dff = self.cfg.d_ff
        b = self.cfg.bytes_act
        return B * (d + dff + dff + d) * b

    def layer_kv_cache_bytes(self, B: int, L: int) -> float:
        d_kv = self.cfg.d_kv
        kv_b = self.cfg.kv_bytes
        kv_data_bytes = 2 * B * L * d_kv * kv_b
        if self.cfg.kv_bits >= 16:
            return kv_data_bytes

        # KiVi-style metadata overhead: every G quantized KV elements carry 4 extra bytes.
        kv_elements = 2 * B * L * d_kv
        num_groups = ceil(kv_elements / self.cfg.kv_group_size)
        kv_meta_bytes = num_groups * self.cfg.kv_meta_bytes_per_group
        return kv_data_bytes + kv_meta_bytes

    def layer_qk_flops(self, B: int, L: int) -> tuple:
        """
        Returns mixed-precision QK flops as (fp16_ops, lowbit_ops).
        - fp16_ops: operations with both Q and K in fp16
        - lowbit_ops: operations with Q in fp16 and K in kv_bits
        """
        dot_terms = B * self.cfg.n_heads * L * self.cfg.head_dim
        # Q operand is always fp16: contributes 2 ops per dot term
        # K operand is in fp16: contributes 2 ops per dot term
        
        if self.cfg.kv_dequant_opt_policy == "ideal":
            # Ideal case: low-bit K can be processed with same throughput as fp16
            fp16_ops = 0 #2 * dot_terms
            lowbit_ops = 0
        elif self.cfg.kv_dequant_opt_policy == "none-quant":
            fp16_ops = 2 * dot_terms
            lowbit_ops = 0 
        elif self.cfg.kv_dequant_opt_policy == "cooperative":
            fp16_ops = 0
            lowbit_ops = 2 * dot_terms 
        else: # "none-opt"
            fp16_ops = (2 + 4) * dot_terms 
            lowbit_ops = 0

        return (fp16_ops, lowbit_ops)

    def layer_qk_dot_terms(self, B: int, L: int) -> float:
        return B * self.cfg.n_heads * L * self.cfg.head_dim

    def layer_av_flops(self, B: int, L: int) -> float:
        if self.cfg.kv_bits < 16: # dequantization overhead applies to AV when K is in low-bit format and no dequant opt policy is in place
            return 4 * B * self.cfg.n_heads * L * self.cfg.head_dim
            # If K is in 2-bit or 16+ bit, we can do AV with same throughput as fp16
        else:
            return 2 * B * self.cfg.n_heads * L * self.cfg.head_dim

    def layer_softmax_flops(self, B: int, L: int) -> float:
        return 5 * B * self.cfg.n_heads * L

    def layer_attn_flops(self, B: int, L: int) -> tuple:
        """
        Returns mixed-precision attention flops as (fp16_ops, lowbit_ops).
        Combines QK, AV, and softmax operations.
        """
        qk_fp16, qk_lowbit = self.layer_qk_flops(B, L)
        av_flops = self.layer_av_flops(B, L)  # AV is always fp16
        softmax_flops = self.layer_softmax_flops(B, L)  # softmax is always fp16
        
        fp16_ops = qk_fp16 + av_flops + softmax_flops
        lowbit_ops = qk_lowbit  # Only QK has lowbit component
        return (fp16_ops, lowbit_ops)

    def layer_attn_bytes_gpu(self, B: int, L: int) -> float:
        score_bytes = B * self.cfg.n_heads * L * self.cfg.bytes_act
        out_bytes = B * self.cfg.d_model * self.cfg.bytes_act
        return self.layer_kv_cache_bytes(B, L) + score_bytes + out_bytes

    def layer_attn_pim_components(self, B: int, L: int) -> Dict[str, float]:
        q_bytes_per_seq = self.cfg.d_model * self.cfg.bytes_act
        kv_total = self.layer_kv_cache_bytes(B, L)
        score_bytes = B * self.cfg.n_heads * L * self.cfg.bytes_act
        score_reduce_bytes = B * self.cfg.n_heads * self.cfg.bytes_act * 4
        out_reduce_bytes = B * self.cfg.d_model * self.cfg.bytes_act
        handoff_bytes = B * self.cfg.d_model * self.cfg.bytes_act

        # Get mixed-precision QK flops
        qk_fp16, qk_lowbit = self.layer_qk_flops(B, L)
        
        return {
            "q_bytes_per_seq": q_bytes_per_seq,
            "qk_bytes": kv_total * 0.5,
            "qk_fp16_flops": qk_fp16,
            "qk_lowbit_flops": qk_lowbit,
            "softmax_flops": self.layer_softmax_flops(B, L),
            "score_bytes": score_bytes,
            "score_reduce_bytes": score_reduce_bytes,
            "av_bytes": kv_total * 0.5,
            "av_flops": self.layer_av_flops(B, L),
            "out_reduce_bytes": out_reduce_bytes,
            "handoff_bytes": handoff_bytes,
            "qk_to_gpu_handoff_bytes": score_bytes,
        }


# ============================================================
# Simulator
# ============================================================

class LLMInferenceSimulator:
    OP_NAMES = [
        "qkv_proj",
        "attn_qk",
        "attn_softmax_reduce",
        "attn_av",
        "attn_handoff",
        "mlp",
    ]

    def __init__(self, model: LLMConfig, gpu: A100GPU, pim: NearBankPIMDevice):
        self.model = model
        self.gpu = gpu
        self.pim = pim
        self.cost = LLMDecodeCostModel(model)

    def _empty_op_accumulator(self) -> Dict[str, float]:
        return {name: 0.0 for name in self.OP_NAMES}

    def _gpu_qk_ops(self, B: int, L: int) -> tuple:
        """Return (fp16_ops, lowbit_ops) for QK computation on GPU"""
        return self.cost.layer_qk_flops(B, L)

    def _pim_qk_ops(self, B: int, L: int) -> tuple:
        """Return (fp16_ops, lowbit_ops) for QK computation on PIM"""
        return self.cost.layer_qk_flops(B, L) 

    def _av_ops(self, B: int, L: int) -> float:
        if self.model.kv_bits == 2 or self.model.kv_bits >= 16:
            return self.cost.layer_av_flops(B, L)

        return self.cost.layer_av_flops(B, L) 

    def _one_layer_gpu_only_ops_s(self, B: int, L: int) -> Dict[str, float]:
        qkv_proj = self.gpu.roofline_time_s(
            self.cost.layer_proj_bytes(B),
            self.cost.layer_proj_flops(B),
            launches=2,
        )

        # Get QK operations (fp16_ops, lowbit_ops)
        qk_fp16, qk_lowbit = self._gpu_qk_ops(B, L)
        
        qk_gpu = self.gpu.roofline_components_s(
            bytes_moved=0.5 * self.cost.layer_attn_bytes_gpu(B, L),
            flops_fp16=qk_fp16,
            flops_lowbit=qk_lowbit,
            kv_bits=self.model.kv_bits,
            launches=1,
        )
        attn_qk = qk_gpu["t_exec_s"]
        
        # Breakdown for attn_qk: show actual bottleneck
        qk_mem = qk_gpu["t_mem_s"]
        qk_comp = qk_gpu["t_cmp_s"]
        qk_launch = qk_gpu["t_launch_s"]
        qk_dominant = max(qk_mem, qk_comp)
        qk_others = qk_launch + abs(qk_mem - qk_comp)  # Other component is the non-dominant one plus launch

        attn_softmax_reduce = self.gpu.roofline_time_s(
            bytes_moved=0.1 * self.cost.layer_attn_bytes_gpu(B, L),
            flops_fp16=self.cost.layer_softmax_flops(B, L),
            launches=1,
        )

        av_flops = self._av_ops(B, L)
        av_gpu = self.gpu.roofline_components_s(
            bytes_moved=0.4 * self.cost.layer_attn_bytes_gpu(B, L),
            flops_fp16=av_flops,
            launches=1,
        )
        attn_av = av_gpu["t_exec_s"]
        
        # Breakdown for attn_av: show actual bottleneck
        av_mem = av_gpu["t_mem_s"]
        av_comp = av_gpu["t_cmp_s"]
        av_launch = av_gpu["t_launch_s"]
        av_dominant = max(av_mem, av_comp)
        av_others = av_launch + abs(av_mem - av_comp)  # Other component is the non-dominant one plus launch

        attn_handoff = 0.0

        mlp = self.gpu.roofline_time_s(
            self.cost.layer_mlp_bytes(B),
            self.cost.layer_mlp_flops(B),
            launches=2,
        )

        return {
            "qkv_proj": qkv_proj,
            "attn_qk": attn_qk,
            "attn_softmax_reduce": attn_softmax_reduce,
            "attn_av": attn_av,
            "attn_handoff": attn_handoff,
            "mlp": mlp,
            "_gpu_attn_qk_breakdown": {
                "memory_access_s": qk_mem,
                "computation_s": qk_comp,
                "others_s": qk_launch,
            },
            "_gpu_attn_av_breakdown": {
                "memory_access_s": av_mem,
                "computation_s": av_comp,
                "others_s": av_launch,
            },
        }

    def _one_layer_hybrid_ops_s(self, B: int, L: int, force_all_attn_on_pim: bool = False) -> Dict[str, float]:
        c = self.cost.layer_attn_pim_components(B, L)
        kv_bits = self.model.kv_bits
        quantized_kv = kv_bits < 16
        use_quantized_split = quantized_kv and (not force_all_attn_on_pim)
        qk_fp16, qk_lowbit = self._pim_qk_ops(B, L)
        av_ops = self._av_ops(B, L)

        qkv_proj = self.gpu.roofline_time_s(
            self.cost.layer_proj_bytes(B),
            self.cost.layer_proj_flops(B),
            launches=2,
        )
        
        # attn_qk breakdown: broadcast + data movement + computation + score reduction
        # Score reduction belongs here: after per-bank Q·K^T, banks must exchange
        # partial score statistics (max/sum) before a globally consistent softmax can run.
        qk_roof = self.pim.roofline_components_s(
            c["qk_bytes"], 
            flops_fp16=qk_fp16,
            flops_lowbit=qk_lowbit,
            kv_bits=kv_bits
        )
        qk_bcast = self.pim.broadcast_time_s(c["q_bytes_per_seq"], batch_size=B, use_host_link=False)
        qk_score_reduce = self.pim.reduction_time_s(c["score_reduce_bytes"])
        qk_sync = self.pim.sync_time_s()
        attn_qk = qk_bcast + qk_roof["t_exec_s"] + qk_score_reduce + qk_sync

        # Q broadcast traffic. "source" is bytes injected once; "fanout" is
        # aggregate network movement and should align with broadcast_time_s.
        qk_bcast_source_traffic_bytes = c["q_bytes_per_seq"] * B
        #qk_bcast_fanout_traffic_bytes = self.pim.total_active_banks() * qk_bcast_source_traffic_bytes
        qk_bcast_fanout_traffic_bytes = qk_bcast_source_traffic_bytes
        
        # K-cache read traffic is the bytes consumed by the QK roofline memory path.
        qk_kcache_read_traffic_bytes = c["qk_bytes"]

        # For attn_qk breakdown: memory includes broadcast and data loading;
        # reduction is the cross-bank score statistics aggregation
        qk_mem = qk_bcast + qk_roof["t_mem_s"]
        qk_reduce = qk_score_reduce
        qk_comp = qk_roof["t_cmp_s"]
        qk_others = qk_sync

        if use_quantized_split:
            # Quantized KV execution mode:
            # 1) QK on PIM, 2) softmax on GPU, 3) AV (including V dequant) on GPU.
            # Handoff moves QK scores from PIM to GPU for downstream attention ops.
            softmax_compute = self.gpu.roofline_time_s(
                bytes_moved=0.1 * self.cost.layer_attn_bytes_gpu(B, L),
                flops_fp16=self.cost.layer_softmax_flops(B, L),
                launches=1,
            )
            attn_softmax_reduce = softmax_compute

            av_gpu = self.gpu.roofline_components_s(
                bytes_moved=0.4 * self.cost.layer_attn_bytes_gpu(B, L),
                flops_fp16=av_ops,
                launches=1,
            )
            attn_av = av_gpu["t_exec_s"]

            # No PIM-side AV or AV reduction in this mode.
            av_vcache_read_traffic_bytes = 0.0
            av_out_reduce_traffic_bytes = 0.0
            av_mem = 0.0
            av_reduce_time = 0.0
            av_comp = 0.0
            av_others = 0.0

            attn_handoff = self.pim.handoff_time_s(c["qk_to_gpu_handoff_bytes"])
        else:
            # Baseline hybrid mode: softmax/AV remain on PIM.
            softmax_compute = self.pim.roofline_time_s(
                0.05 * c["qk_bytes"], 
                flops_fp16=c["softmax_flops"],
                kv_bits=16
            )
            attn_softmax_reduce = softmax_compute

            # attn_av breakdown: memory access + reduction + computation + sync
            av_roof = self.pim.roofline_components_s(
                c["av_bytes"], 
                flops_fp16=av_ops,
                kv_bits=kv_bits
            )
            av_reduce = self.pim.reduction_time_s(c["out_reduce_bytes"])
            av_sync = self.pim.sync_time_s()
            attn_av = av_roof["t_exec_s"] + av_reduce + av_sync

            # attn_av traffic breakdown
            av_vcache_read_traffic_bytes = c["av_bytes"]
            av_out_reduce_traffic_bytes = c["out_reduce_bytes"]
            
            # For attn_av: separate out memory, reduction, computation, and sync
            av_mem = av_roof["t_mem_s"]
            av_reduce_time = av_reduce
            av_comp = av_roof["t_cmp_s"]
            av_others = av_sync

            attn_handoff = self.pim.handoff_time_s(c["handoff_bytes"])

        mlp = self.gpu.roofline_time_s(
            self.cost.layer_mlp_bytes(B),
            self.cost.layer_mlp_flops(B),
            launches=2,
        )

        return {
            "qkv_proj": qkv_proj,
            "attn_qk": attn_qk,
            "attn_softmax_reduce": attn_softmax_reduce,
            "attn_av": attn_av,
            "attn_handoff": attn_handoff,
            "mlp": mlp,
            "_pim_attn_qk_breakdown": {
                "memory_access_s": qk_mem,
                "reduction_s": qk_reduce,
                "computation_s": qk_comp,
                "others_s": qk_others,
                "broadcast_source_traffic_bytes": qk_bcast_source_traffic_bytes,
                "broadcast_fanout_traffic_bytes": qk_bcast_fanout_traffic_bytes,
                "k_cache_read_traffic_bytes": qk_kcache_read_traffic_bytes,
            },
            "_pim_attn_av_breakdown": {
                "memory_access_s": av_mem,
                "reduction_s": av_reduce_time,
                "computation_s": av_comp,
                "others_s": av_others,
                "v_cache_read_traffic_bytes": av_vcache_read_traffic_bytes,
                "out_reduce_traffic_bytes": av_out_reduce_traffic_bytes,
            },
        }

    def simulate_decode(self, req: DecodeRequest, mode: str = "gpu_only") -> Dict[str, object]:
        if mode not in {"gpu_only", "hybrid_attn_on_pim", "hybrid_attn_on_pim_original"}:
            raise ValueError("mode must be 'gpu_only', 'hybrid_attn_on_pim', or 'hybrid_attn_on_pim_original'.")

        total_ops = self._empty_op_accumulator()
        
        # GPU attention breakdown accumulators
        gpu_attn_qk_breakdown = {
            "memory_access_s": 0.0,
            "computation_s": 0.0,
            "others_s": 0.0,
        }
        gpu_attn_av_breakdown = {
            "memory_access_s": 0.0,
            "computation_s": 0.0,
            "others_s": 0.0,
        }
        
        # PIM attention breakdown accumulators
        pim_attn_qk_breakdown = {
            "memory_access_s": 0.0,
            "reduction_s": 0.0,
            "computation_s": 0.0,
            "others_s": 0.0,
            "broadcast_source_traffic_bytes": 0.0,
            "broadcast_fanout_traffic_bytes": 0.0,
            "k_cache_read_traffic_bytes": 0.0,
        }
        pim_attn_av_breakdown = {
            "memory_access_s": 0.0,
            "reduction_s": 0.0,
            "computation_s": 0.0,
            "others_s": 0.0,
            "v_cache_read_traffic_bytes": 0.0,
            "out_reduce_traffic_bytes": 0.0,
        }
        
        per_step: List[Dict[str, float]] = []

        for step in range(req.gen_len):
            L = req.context_len + step

            if mode == "gpu_only":
                one_layer = self._one_layer_gpu_only_ops_s(req.batch_size, L)
            elif mode == "hybrid_attn_on_pim_original":
                one_layer = self._one_layer_hybrid_ops_s(req.batch_size, L, force_all_attn_on_pim=True)
            else:
                one_layer = self._one_layer_hybrid_ops_s(req.batch_size, L)

            gpu_qk_bd = one_layer.pop("_gpu_attn_qk_breakdown", None)
            gpu_av_bd = one_layer.pop("_gpu_attn_av_breakdown", None)
            pim_qk_bd = one_layer.pop("_pim_attn_qk_breakdown", None)
            pim_av_bd = one_layer.pop("_pim_attn_av_breakdown", None)

            step_ops = {op: t * self.model.n_layers for op, t in one_layer.items()}
            step_total = sum(step_ops.values())

            for op, t in step_ops.items():
                total_ops[op] += t

            if gpu_qk_bd is not None:
                for k, v in gpu_qk_bd.items():
                    gpu_attn_qk_breakdown[k] += v * self.model.n_layers
            
            if gpu_av_bd is not None:
                for k, v in gpu_av_bd.items():
                    gpu_attn_av_breakdown[k] += v * self.model.n_layers

            if pim_qk_bd is not None:
                for k, v in pim_qk_bd.items():
                    pim_attn_qk_breakdown[k] += v * self.model.n_layers
            
            if pim_av_bd is not None:
                for k, v in pim_av_bd.items():
                    pim_attn_av_breakdown[k] += v * self.model.n_layers

            per_step.append({
                "step": step,
                "seq_len": L,
                **step_ops,
                "total_s": step_total,
            })

        total_time_s = sum(total_ops.values())
        total_tokens = req.batch_size * req.gen_len
        throughput_tok_s = total_tokens / total_time_s if total_time_s > 0 else 0.0

        breakdown_rows = []
        for op in self.OP_NAMES:
            total_ms = total_ops[op] * 1e3
            avg_ms = total_ms / req.gen_len if req.gen_len > 0 else 0.0
            share = total_ops[op] / total_time_s if total_time_s > 0 else 0.0
            breakdown_rows.append({
                "op_type": op,
                "count": self.model.n_layers * req.gen_len,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
                "share": share,
            })

        return {
            "model": self.model.name,
            "kv_bits": self.model.kv_bits,
            "mode": mode,
            "batch_size": req.batch_size,
            "context_len": req.context_len,
            "gen_len": req.gen_len,
            "total_generated_tokens": total_tokens,
            "total_time_s": total_time_s,
            "throughput_tok_s": throughput_tok_s,
            "latency_per_generated_token_s": total_time_s / total_tokens if total_tokens > 0 else 0.0,
            "breakdown_rows": breakdown_rows,
            "breakdown_totals_s": total_ops,
            "gpu_attn_qk_breakdown": gpu_attn_qk_breakdown,
            "gpu_attn_av_breakdown": gpu_attn_av_breakdown,
            "pim_attn_qk_breakdown": pim_attn_qk_breakdown,
            "pim_attn_av_breakdown": pim_attn_av_breakdown,
            "per_step": per_step,
        }

    @staticmethod
    def print_breakdown_table(result: Dict[str, object]) -> None:
        rows = result["breakdown_rows"]

        print(
            f"Model: {result['model']} | Mode: {result['mode']} | "
            f"Batch: {result['batch_size']} | Context: {result['context_len']} | Gen: {result['gen_len']}"
        )
        print("Operator latency breakdown:")
        print(f"{'op_type':>24} {'count':>10} {'total_ms':>12} {'avg_ms':>12} {'share':>10}")
        print("-" * 74)
        for r in rows:
            print(
                f"{r['op_type']:>24} "
                f"{r['count']:>10d} "
                f"{r['total_ms']:>12.3f} "
                f"{r['avg_ms']:>12.3f} "
                f"{100.0 * r['share']:>9.2f}%"
            )
        print("-" * 74)
        print(
            f"{'TOTAL':>24} "
            f"{sum(r['count'] for r in rows):>10d} "
            f"{sum(r['total_ms'] for r in rows):>12.3f} "
            f"{(sum(r['total_ms'] for r in rows) / result['gen_len']):>12.3f} "
            f"{100.0:>9.2f}%"
        )
        print(f"Throughput: {result['throughput_tok_s']:.3f} tok/s")
        print(f"Latency/token: {1e3 * result['latency_per_generated_token_s']:.3f} ms")

        # GPU mode attention breakdown
        if result.get("mode") == "gpu_only":
            gpu_qk_bd = result.get("gpu_attn_qk_breakdown", {})
            gpu_av_bd = result.get("gpu_attn_av_breakdown", {})
            
            print("\n=== GPU Attention Performance Breakdown ===")
            print("(Note: Memory and Compute times may overlap due to pipelining)")
            
            if gpu_qk_bd and any(gpu_qk_bd.values()):
                print("\nattn_qk (Query-Key):")
                print(f"{'Component':>20} {'Time (ms)':>15} {'Relative':>12}")
                print("-" * 50)
                
                mem_ms = gpu_qk_bd.get("memory_access_s", 0.0) * 1e3
                comp_ms = gpu_qk_bd.get("computation_s", 0.0) * 1e3
                others_ms = gpu_qk_bd.get("others_s", 0.0) * 1e3
                bottleneck_ms = max(mem_ms, comp_ms) + others_ms
                
                if bottleneck_ms > 0:
                    print(f"{'Memory Access':>20} {mem_ms:>15.3f} {100.0 * mem_ms / max(mem_ms, comp_ms):>11.2f}%")
                    print(f"{'Computation':>20} {comp_ms:>15.3f} {100.0 * comp_ms / max(mem_ms, comp_ms):>11.2f}%")
                    print(f"{'Launch Overhead':>20} {others_ms:>15.3f} {'N/A':>12}")
                    print("-" * 50)
                    print(f"{'Bottleneck':>20} {bottleneck_ms:>15.3f} {'N/A':>12}")
            
            if gpu_av_bd and any(gpu_av_bd.values()):
                print("\nattn_av (Attention-Value):")
                print(f"{'Component':>20} {'Time (ms)':>15} {'Relative':>12}")
                print("-" * 50)
                
                mem_ms = gpu_av_bd.get("memory_access_s", 0.0) * 1e3
                comp_ms = gpu_av_bd.get("computation_s", 0.0) * 1e3
                others_ms = gpu_av_bd.get("others_s", 0.0) * 1e3
                bottleneck_ms = max(mem_ms, comp_ms) + others_ms
                
                if bottleneck_ms > 0:
                    print(f"{'Memory Access':>20} {mem_ms:>15.3f} {100.0 * mem_ms / max(mem_ms, comp_ms):>11.2f}%")
                    print(f"{'Computation':>20} {comp_ms:>15.3f} {100.0 * comp_ms / max(mem_ms, comp_ms):>11.2f}%")
                    print(f"{'Launch Overhead':>20} {others_ms:>15.3f} {'N/A':>12}")
                    print("-" * 50)
                    print(f"{'Bottleneck':>20} {bottleneck_ms:>15.3f} {'N/A':>12}")

        # PIM mode attention breakdown
        elif result.get("mode") in {"hybrid_attn_on_pim", "hybrid_attn_on_pim_original"}:
            pim_qk_bd = result.get("pim_attn_qk_breakdown", {})
            pim_av_bd = result.get("pim_attn_av_breakdown", {})
            
            print("\n=== PIM Attention Performance Breakdown ===")
            if result.get("kv_bits", 16) < 16 and result.get("mode") == "hybrid_attn_on_pim":
                print("Execution map (quantized KV): QxK^T on PIM; softmax + AV (+V dequant) on GPU.")
            elif result.get("kv_bits", 16) < 16 and result.get("mode") == "hybrid_attn_on_pim_original":
                print("Execution map (original GPU-PIM): QxK^T + softmax + AV on PIM.")
            
            if pim_qk_bd and any(pim_qk_bd.values()):
                print("\nattn_qk (Query-Key):")
                print(f"{'Component':>20} {'Time (ms)':>15} {'Share':>12}")
                print("-" * 50)
                
                mem_ms = pim_qk_bd.get("memory_access_s", 0.0) * 1e3
                red_ms = pim_qk_bd.get("reduction_s", 0.0) * 1e3
                comp_ms = pim_qk_bd.get("computation_s", 0.0) * 1e3
                others_ms = pim_qk_bd.get("others_s", 0.0) * 1e3
                qk_total_ms = mem_ms + red_ms + comp_ms + others_ms
                
                if qk_total_ms > 0:
                    print(f"{'Memory Access':>20} {mem_ms:>15.3f} {100.0 * mem_ms / qk_total_ms:>11.2f}%")
                    print(f"{'Reduction':>20} {red_ms:>15.3f} {100.0 * red_ms / qk_total_ms:>11.2f}%")
                    print(f"{'Computation':>20} {comp_ms:>15.3f} {100.0 * comp_ms / qk_total_ms:>11.2f}%")
                    print(f"{'Others (sync)':>20} {others_ms:>15.3f} {100.0 * others_ms / qk_total_ms:>11.2f}%")
                    print("-" * 50)
                    print(f"{'TOTAL':>20} {qk_total_ms:>15.3f} {100.0:>11.2f}%")

                    bcast_src_gb = pim_qk_bd.get("broadcast_source_traffic_bytes", 0.0) / 1e9
                    bcast_fanout_gb = pim_qk_bd.get("broadcast_fanout_traffic_bytes", 0.0) / 1e9
                    kread_gb = pim_qk_bd.get("k_cache_read_traffic_bytes", 0.0) / 1e9
                    print("\nattn_qk traffic:")
                    print(f"{'Type':>20} {'Traffic (GB)':>15}")
                    print("-" * 36)
                    print(f"{'Q broadcast(src)':>20} {bcast_src_gb:>15.3f}")
                    print(f"{'Q broadcast(fanout)':>20} {bcast_fanout_gb:>15.3f}")
                    print(f"{'K-cache read':>20} {kread_gb:>15.3f}")
            
            if pim_av_bd and any(pim_av_bd.values()):
                print("\nattn_av (Attention-Value):")
                print(f"{'Component':>20} {'Time (ms)':>15} {'Share':>12}")
                print("-" * 50)
                
                mem_ms = pim_av_bd.get("memory_access_s", 0.0) * 1e3
                red_ms = pim_av_bd.get("reduction_s", 0.0) * 1e3
                comp_ms = pim_av_bd.get("computation_s", 0.0) * 1e3
                others_ms = pim_av_bd.get("others_s", 0.0) * 1e3
                av_total_ms = mem_ms + red_ms + comp_ms + others_ms
                
                if av_total_ms > 0:
                    print(f"{'Memory Access':>20} {mem_ms:>15.3f} {100.0 * mem_ms / av_total_ms:>11.2f}%")
                    print(f"{'Reduction':>20} {red_ms:>15.3f} {100.0 * red_ms / av_total_ms:>11.2f}%")
                    print(f"{'Computation':>20} {comp_ms:>15.3f} {100.0 * comp_ms / av_total_ms:>11.2f}%")
                    print(f"{'Others (sync)':>20} {others_ms:>15.3f} {100.0 * others_ms / av_total_ms:>11.2f}%")
                    print("-" * 50)
                    print(f"{'TOTAL':>20} {av_total_ms:>15.3f} {100.0:>11.2f}%")

                    vread_gb = pim_av_bd.get("v_cache_read_traffic_bytes", 0.0) / 1e9
                    out_reduce_gb = pim_av_bd.get("out_reduce_traffic_bytes", 0.0) / 1e9
                    print("\nattn_av traffic:")
                    print(f"{'Type':>20} {'Traffic (GB)':>15}")
                    print("-" * 36)
                    print(f"{'V-cache read':>20} {vread_gb:>15.3f}")
                    print(f"{'Output reduction':>20} {out_reduce_gb:>15.3f}")


# ============================================================
# Example configs
# ============================================================

def llama3_8b_config(kv_bits: int = 16) -> LLMConfig:
    return LLMConfig(
        name=f"Llama-3-8B-kv{kv_bits}",
        n_layers=32,
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        ffn_mult=3.5,
        bytes_act=2,
        kv_bits=kv_bits,
    )


def llama2_70b_config(kv_bits: int = 16) -> LLMConfig:
    return LLMConfig(
        name=f"Llama-2-70B-kv{kv_bits}",
        n_layers=80,
        hidden_size=8192,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        ffn_mult=3.5,
        bytes_act=2,
        kv_bits=kv_bits,
    )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    gpu = A100GPU(
        model_name="A100 80GB (PCIe)",
        memory_type="HBM2e",
        mem_bw_GBs=1935.0,
        tensor_flops_GFs=312_000.0,
        bf16_tensor_tflops_dense=312.0,
        bf16_tensor_tflops_sparse=624.0,
        fp16_tensor_tflops_dense=312.0,
        fp16_tensor_tflops_sparse=624.0,
        int8_tensor_tops_dense=624.0,
        int8_tensor_tops_sparse=1248.0,
        int4_tensor_tops_dense=1248.0,
        int4_tensor_tops_sparse=2496.0,
        mem_eff=0.85,
        compute_eff=0.35,
        kernel_launch_us=3.0,
    )

    pim = NearBankPIMDevice(
        n_stacks=5,
        banks_per_stack=256,
        active_banks_per_stack=256,
        internal_bank_bw_GBs=16.0,
        macs_per_pe= 16,
        pe_frequency_GHz=1.0,
        ops_per_mac_per_cycle=2,
        bw_eff=0.85,
        flop_eff=0.80,
        hbm_stack_host_bw_GBs=819.2,
        reduce_bw_GBs=512.0,
        reduce_latency_ns=15.0,
        sync_latency_ns=40.0,
    )

    #model = llama2_70b_config(kv_bits=16)
    model = llama3_8b_config(kv_bits=2)
    
    req = DecodeRequest(batch_size=64, context_len=16400, gen_len=128)

    sim = LLMInferenceSimulator(model, gpu, pim)

    print("\n=== Hardware Configuration ===")
    print("GPU summary:")
    for key, value in gpu.summary().items():
        print(f"  {key}: {value}")
    print("PIM summary:")
    for key, value in pim.summary().items():
        print(f"  {key}: {value}")
    print(
        "PIM overall (effective): "
        f"BW={pim.local_bw_GBs():.2f} GB/s, "
        f"Compute={pim.local_flops_GFs():.2f} GFLOP/s"
    )

    gpu_only = sim.simulate_decode(req, mode="gpu_only")
    original_gpu_pim = sim.simulate_decode(req, mode="hybrid_attn_on_pim_original")
    hybrid_attn = sim.simulate_decode(req, mode="hybrid_attn_on_pim")

    print("\n=== GPU-only ===")
    sim.print_breakdown_table(gpu_only)

    print("\n=== Original GPU-PIM (all attention ops on PIM) ===")
    sim.print_breakdown_table(original_gpu_pim)

    print("\n=== Hybrid attention (QK on PIM, softmax+AV on GPU for quantized KV) ===")
    sim.print_breakdown_table(hybrid_attn)