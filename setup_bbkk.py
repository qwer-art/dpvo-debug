import os
import os.path as osp
from setuptools import setup, find_packages

conda_prefix = '/home/jerett/anaconda3/envs/dpvo_fixed'
os.environ['CUDA_HOME'] = conda_prefix

# 绕过 CUDA 版本检查
os.environ['TORCH_CUDA_VERSION'] = '11.8'
os.environ['FORCE_CUDA'] = '1'

# 猴子补丁：修改 PyTorch 的版本检查逻辑
import torch.utils.cpp_extension as ext

original_check_cuda_version = ext._check_cuda_version

def patched_check_cuda_version(compiler_name, compiler_version):
    print("跳过 CUDA 版本检查...")
    return True  # 总是返回成功

ext._check_cuda_version = patched_check_cuda_version

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    print(f"使用 PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
except ImportError:
    raise RuntimeError("Torch not found")

ROOT = osp.dirname(osp.abspath(__file__))

# 您的 nvcc_args 和其余代码保持不变...
nvcc_args = [
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-O3'
]

setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': nvcc_args,
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': nvcc_args,
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp',
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': nvcc_args}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })