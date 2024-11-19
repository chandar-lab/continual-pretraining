# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import subprocess


def _get_cuda_bare_metal_version(cuda_dir=None):
    # Check for hipcc in ROCm environment
    compiler = f"{cuda_dir}/bin/hipcc" if cuda_dir else None
    print("compiler",compiler)
    if not compiler:
        raise EnvironmentError("CUDA_HOME or ROCM_PATH not set correctly.")
    
    try:
        raw_output = subprocess.check_output([compiler, "--version"], universal_newlines=True)
        output = raw_output.split()
        version_idx = output.index("version:") + 1
        release = output[version_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
        return raw_output, bare_metal_major, bare_metal_minor
    except (FileNotFoundError, IndexError) as e:
        raise EnvironmentError(f"Failed to determine version using {compiler}. Ensure ROCm is installed correctly.") from e



srcpath = Path(__file__).parent.absolute()
cc_flag = []
_, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")

nvcc_flags = [
    "-O3",
    "-gencode",
    "arch=compute_70,code=sm_70",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]
cuda_ext_args = {"cxx": ["-O3"], "nvcc": nvcc_flags + cc_flag}
layernorm_cuda_args = {
    "cxx": ["-O3"],
    "nvcc": nvcc_flags + cc_flag + ["-maxrregcount=50"],
}
setup(
    name="fused_kernels",
    version="0.0.2",
    author="EleutherAI",
    author_email="contact@eleuther.ai",
    include_package_data=False,
    ext_modules=[
        CUDAExtension(
            name="scaled_upper_triang_masked_softmax_cuda",
            sources=[
                str(srcpath / "scaled_upper_triang_masked_softmax.cpp"),
                str(srcpath / "scaled_upper_triang_masked_softmax_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        ),
        CUDAExtension(
            name="scaled_masked_softmax_cuda",
            sources=[
                str(srcpath / "scaled_masked_softmax.cpp"),
                str(srcpath / "scaled_masked_softmax_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        ),
        CUDAExtension(
            name="fused_rotary_positional_embedding",
            sources=[
                str(srcpath / "fused_rotary_positional_embedding.cpp"),
                str(srcpath / "fused_rotary_positional_embedding_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)