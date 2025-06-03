# Check if HIP is installed
which hipcc
hipcc --version

# Check if HIP headers exist
ls -la /opt/rocm/include/hip/
find /opt/rocm -name "hip_runtime.h" 2>/dev/null

export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
export C_INCLUDE_PATH=/opt/rocm/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/rocm/include:$CPLUS_INCLUDE_PATH

# Check environment variables
echo $ROCM_PATH
echo $HIP_PATH
echo $PATH

git clone -b amd-staging https://github.com/ROCm/rocm-examples.git
cd rocm-examples
export ROCM_GPU=$(rocminfo | grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/')
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DGPU_TARGETS=$ROCM_GPU
make rocblas_level_1_axpy -j$(nproc)
cd Libraries/rocBLAS/level_1/axpy && ./axpy

hipcc \
    -I../../../../Common \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lrocblas \
    -lhip \
    --offload-arch=gfx1032 \
    main.cpp -o axpy

./axpy
