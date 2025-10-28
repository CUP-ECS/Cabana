# Inclduing mpi-advance in spack environment
# -DMPI_ADVANCE_DIR=~/installed-libraries/mpi_advance/
cmake -DCabana_ENABLE_TESTING=ON -DCabana_REQUIRE_MPI=ON -DCabana_REQUIRE_LOCALITY_AWARE=ON -DCabana_REQUIRE_CUDA=ON -DCabana_ENABLE_GRID=ON -DCabana_ENABLE_EXAMPLES=ON -DCabana_REQUIRE_SERIAL=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ..