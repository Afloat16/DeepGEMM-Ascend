__version__ = '0.1.0'

import os
import torch

# Set default environment variables captured when the package was built.
try:
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Ascend toolkit path discovery
ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH',
    os.environ.get('ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest'))


def get_library_root_path():
    """Get the root path of the deep_gemm package (where include/ lives)."""
    return os.path.dirname(os.path.realpath(__file__))


from . import _C
from ._C import (
    set_num_sms,
    get_num_sms,
    set_npu_arch,
    get_npu_arch,
    set_mk_alignment_for_contiguous_layout,
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
    use_deterministic_algorithms,
    get_deterministic_algorithms,
)

# Upstream cuBLASLt API aliases backed by ACLNN
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
)

# DeepGEMM Kernels
from ._C import (
    # FP8 FP4 GEMMs
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    # FP8 GEMMs
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    # ACLNN reference GEMMs
    aclnn_fp8_fp4_gemm_nt, aclnn_fp8_fp4_gemm_nn,
    aclnn_fp8_fp4_gemm_tn, aclnn_fp8_fp4_gemm_tt,
    aclnn_bf16_gemm_nt, aclnn_bf16_gemm_nn,
    aclnn_bf16_gemm_tn, aclnn_bf16_gemm_tt,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    k_grouped_bf16_gemm_tn_contiguous,
    # Einsum kernels
    einsum,
    fp8_einsum,
    # Attention kernels
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_fp4_paged_mqa_logits,
    # Layout kernels
    transform_sf_into_required_layout,
    transform_k_grouped_sf_into_required_layout,
)

from .mega import (
    SymmBuffer,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
)

# Epilogue classes
from ._C import epilogue

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    is_localized,
    destroy_localizer,
)

# TileLang kernel implementations
from .tilelang_ops import tf32_hc_prenorm_gemm

_C.init(__version__, get_library_root_path())
