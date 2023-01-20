# TODO: YOU, THE USER, SHOULD CHANGE THESE TO YOUR DESIRED PATHS
export KOKKOS_INSTALL_DIR=$(spack location -i kokkos)
export CABANA_INSTALL_DIR=$(pwd)/build/install

mkdir -p build/install

cd build
cmake \
    -D CMAKE_BUILD_TYPE="Debug" \
    -D CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_DIR \
    -D CMAKE_INSTALL_PREFIX=$CABANA_INSTALL_DIR \
    -D Cabana_REQUIRE_OPENMP=OFF \
    -D Cabana_ENABLE_EXAMPLES=ON \
    -D Cabana_ENABLE_TESTING=OFF \
    -D Cabana_ENABLE_PERFORMANCE_TESTING=OFF \
    -D Cabana_ENABLE_CAJITA=ON \
    .. ;

make -j 24 install
