from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import numpy as np
import glob
import os

gruppo='gruppo6'

class CustomBuildExt(build_ext):
    def run(self):
        # Compila file NASM prima di build C
        for arch in ['32', '64', '64omp']:
            folder = f"src/{arch}"
            nasm_files = glob.glob(os.path.join(folder, "*.nasm"))
            for nasm_file in nasm_files:
                subprocess.run([
                    'nasm',
                    '-f', 'elf64',
                    '-DPIC',
                    '-I', folder,
                    nasm_file
                ], check=True)

        # Aggiunge i file .o dinamicamente
        for ext in self.extensions:
            if '32' in ext.name:
                ext.extra_objects = glob.glob('src/32/*.o')
            elif '64omp' in ext.name:
                ext.extra_objects = glob.glob('src/64omp/*.o')
            elif '64' in ext.name:
                ext.extra_objects = glob.glob('src/64/*.o')

        super().run()

# ---- FLAGS OTTIMIZZATI PER COMPETIZIONE ----
# Se vuoi usare -march=native solo sul tuo PC:
#   export NATIVE=1
use_native = os.environ.get("NATIVE", "0") == "1"

# Se vuoi usare ASM per 32-bit:
#   export USE_ASM_32=1
use_asm_32 = os.environ.get("USE_ASM_32", "0") == "1"

# Se vuoi usare ASM per 64-bit:
#   export USE_ASM_64=1
use_asm_64 = os.environ.get("USE_ASM_64", "0") == "1"

# Se vuoi usare ASM per 64-bit OpenMP:
#   export USE_ASM_OMP=1
use_asm_omp = os.environ.get("USE_ASM_OMP", "0") == "1"

# Flag di base AGGRESSIVE per competizione
base_cflags = [
    '-O3',                    # Ottimizzazione massima
    '-DNDEBUG',              # Rimuove assert
    '-fPIC',                 # Position independent code
    '-ffast-math',           # Ottimizzazioni matematiche aggressive
    '-funroll-loops',        # Unroll loops automatico
    '-fno-signed-zeros',     # Assume zero non ha segno
    '-fno-trapping-math',    # Elimina controlli eccezioni FP
    '-fassociative-math',    # Riordina operazioni FP
    '-freciprocal-math',     # Usa reciproci veloci
]

if use_native:
    base_cflags += ['-march=native', '-mtune=native']  # SEMPRE attivo per competizione sul tuo PC

# Flags per modulo 32
cflags_32 = base_cflags + ['-msse', '-msse2', '-msse3']
if use_asm_32:
    cflags_32 += ['-DUSE_ASM_APPROX', '-DUSE_ASM_EUCLIDEAN', '-DUSE_ASM_LOWER_BOUND']
else:
    # C puro: aggiungi auto-vectorization hints
    cflags_32 += ['-ftree-vectorize', '-ftree-loop-vectorize']

# Flags per modulo 64
cflags_64 = base_cflags + ['-msse', '-mavx', '-mavx2', '-mfma']
if use_asm_64:
    cflags_64 += ['-DUSE_ASM_APPROX', '-DUSE_ASM_EUCLIDEAN', '-DUSE_ASM_LOWER_BOUND']
else:
    cflags_64 += ['-ftree-vectorize', '-ftree-loop-vectorize']

# Flags per modulo 64omp
cflags_64omp = base_cflags + ['-msse', '-mavx', '-mavx2', '-mfma', '-fopenmp']
if use_asm_omp:
    cflags_64omp += ['-DUSE_ASM_APPROX', '-DUSE_ASM_EUCLIDEAN', '-DUSE_ASM_LOWER_BOUND']
else:
    cflags_64omp += ['-ftree-vectorize', '-ftree-loop-vectorize']

module32 = Extension(
    f"{gruppo}.quantpivot32._quantpivot32",
    sources=['src/32/quantpivot32_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=cflags_32,
    extra_link_args=['-z', 'noexecstack']
)

module64 = Extension(
    f"{gruppo}.quantpivot64._quantpivot64",
    sources=['src/64/quantpivot64_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=cflags_64,
    extra_link_args=['-z', 'noexecstack']
)

module64omp = Extension(
    f"{gruppo}.quantpivot64omp._quantpivot64omp",
    sources=['src/64omp/quantpivot64omp_py.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=cflags_64omp,
    extra_link_args=['-z', 'noexecstack', '-fopenmp']
)

setup(
    name=gruppo,
    version='1.0',
    author="LISTA COMPONENTI GRUPPO",
    packages=find_packages(),
    ext_modules=[module32, module64, module64omp],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['numpy'],
    zip_safe=False
)