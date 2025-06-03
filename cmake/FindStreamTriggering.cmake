# Try to find MPI Advance Stream Triggering (https://github.com/mpi-advance/stream-triggering)
# Once done this will define
#  stream-triggering_FOUND       - System has LibFabric
#  stream-triggering_INCLUDE_DIR - The LibFabric include directories
#  stream-triggering_LIBRARY     - The libraries needed to use LibFabric

# search prefix path
set(stream-triggering_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Help cmake to find MPI Advance Stream Triggering (https://github.com/mpi-advance/stream-triggering).")

find_path(stream-triggering_INCLUDE_DIR stream-triggering.h HINTS ${stream-triggering_PREFIX}/include)
find_library(stream-triggering_LIBRARY NAMES stream-triggering HINTS ${stream-triggering_PREFIX}/lib)

# setup found
if (stream-triggering_LIBRARY AND stream-triggering_INCLUDE_DIR)
  set(stream-triggering_FOUND ON)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(stream-triggering DEFAULT_MSG stream-triggering_INCLUDE_DIR stream-triggering_LIBRARY)

mark_as_advanced(stream-triggering_INCLUDE_DIR stream-triggering_LIBRARY)
if(stream-triggering_INCLUDE_DIR AND stream-triggering_LIBRARY)#  AND NOT TARGET SILO::silo)
  add_library(stream-triggering UNKNOWN IMPORTED)
  set_target_properties(stream-triggering PROPERTIES
    IMPORTED_LOCATION ${stream-triggering_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${stream-triggering_INCLUDE_DIR})
  if(stream-triggering_LINK_FLAGS)
    set_property(TARGET stream-triggering APPEND_STRING PROPERTY INTERFACE_LINK_LIBRARIES " ${stream-triggering_LINK_FLAGS}")
  endif()
endif()
