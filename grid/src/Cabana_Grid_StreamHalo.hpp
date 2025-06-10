/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_Halo.hpp
  \brief Multi-node grid stream-based scatter/gather
*/
#ifndef CABANA_GRID_STREAMHALO_HPP
#define CABANA_GRID_STREAMHALO_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Types.hpp>

#include <Cabana_ParameterPack.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <Cuda/Kokkos_Cuda.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{
namespace Experimental
{

//---------------------------------------------------------------------------//
// StreamHalo
// ---------------------------------------------------------------------------//
/*!
  Stream-triggered multiple array halo communication plan for migrating
  shared data between blocks.
*/
template <class ExecutionSpace, class MemorySpace>
class StreamHaloBase : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using halo_type = Cabana::Grid::Halo<memory_space>;

      //! Enqueue operations to pack arrays into a buffer. Calling code must
    //! fence.
    template <class... ArrayViews>
    void enqueuePackBuffer( const Kokkos::View<char*, memory_space>& buffer,
                            const Kokkos::View<int**, memory_space>& steering,
                            ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::StreamHalo::pack_buffer",
            Kokkos::RangePolicy<ExecutionSpace>( _exec_space, 0,
                                                 steering.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                halo_type::packArray(
                    buffer, steering, i,
                    std::integral_constant<std::size_t,
                                           sizeof...( ArrayViews ) - 1>(),
                    pp );
            } );
    }

    //! Enqueue operations to pack all arrays into a buffer. Calling code must
    //! fence. Separate from the code to pack a single buffer to enable a later
    //! version with a fused packing kernel
    template <class... ArrayViews>
    void enqueuePackBuffers(
        const std::vector<Kokkos::View<char*, memory_space>>& buffers,
        const std::vector<Kokkos::View<int**, memory_space>>& steerings,
        ArrayViews... array_views ) const
    {
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for ( int n = 0; n < buffers.size(); n++ )
        {
            Kokkos::View<char*, memory_space> buffer = buffers[n];
            Kokkos::View<int**, memory_space> steering = steerings[n];
            if ( buffers[n].size() > 0 )
                enqueuePackBuffer( buffer, steering, array_views... );
        }
    }

    //! Enqueue operations to unpack buffer into arrays. Calling code must
    //! fence.
    template <class ReduceOp, class... ArrayViews>
    void enqueueUnpackBuffer( const ReduceOp& reduce_op,
                              const Kokkos::View<char*, memory_space>& buffer,
                              const Kokkos::View<int**, memory_space>& steering,
                              ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::StreamHalo::unpack_buffer",
            Kokkos::RangePolicy<ExecutionSpace>( _exec_space, 0,
                                                 steering.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                halo_type::unpackArray(
                    reduce_op, buffer, steering, i,
                    std::integral_constant<std::size_t,
                                           sizeof...( ArrayViews ) - 1>(),
                    pp );
            } );
    }

    //! Enqueue operations to unpack buffer into arrays. Calling code must
    //! fence. Separate from the code to unpack a single buffer to enable a
    //! later version with a fused unpacking kernel
    template <class ReduceOp, class... ArrayViews>
    void enqueueUnpackBuffers(
        const ReduceOp& reduce_op,
        const std::vector<Kokkos::View<char*, memory_space>>& buffers,
        const std::vector<Kokkos::View<int**, memory_space>>& steerings,
        ArrayViews... array_views ) const
    {
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for ( int n = 0; n < buffers.size(); n++ )
        {
            Kokkos::View<char*, memory_space> buffer = buffers[n];
            Kokkos::View<int**, memory_space> steering = steerings[n];
            if ( buffers[n].size() > 0 )
                enqueueUnpackBuffer( reduce_op, buffer, steering,
                                     array_views... );
        }
    }
  protected:
    template <class Pattern, class... ArrayTypes>
    StreamHaloBase( const ExecutionSpace& exec_space, const Pattern& pattern,
                    const int width, const ArrayTypes&... arrays )
        : Cabana::Grid::Halo<MemorySpace>( pattern, width, arrays... )
        , _exec_space( exec_space )
    {
    }

  protected:
    const execution_space
        _exec_space; // not a reference - we want the copy here.
};

template <class ExecutionSpace, class MemorySpace,
          class CommSpace = Cabana::CommSpace::Mpi>
class StreamHalo;

} // namespace Experimental
} // namespace Grid
} // namespace Cabana

// Include further specializations of StreamHalo to get implementations.
#ifdef Cabana_ENABLE_MPI
#include <impl/Cabana_Grid_MpiStreamHalo.hpp>
#endif

#ifdef Cabana_ENABLE_MPICH
#include <impl/Cabana_Grid_MpichStreamHalo.hpp>
#endif // MPICH

#ifdef Cabana_ENABLE_CRAY_MPI
#include <impl/Cabana_Grid_CrayMpiStreamHalo.hpp>
#endif // CRAY_MPI

#ifdef Cabana_ENABLE_MPI_ADVANCE
#include <impl/Cabana_Grid_MpiAdvanceStreamHalo.hpp>
#endif // MPIADVANCE

namespace Cabana
{
namespace Grid
{
namespace Experimental
{

/*!
  \brief Halo creation function.
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param arrays The arrays over which to build the halo.
  \return Shared pointer to a Halo.
*/
template <class ExecutionSpace, class Pattern, class... ArrayTypes>
auto createStreamHalo( const ExecutionSpace& exec_space, const Pattern& pattern,
                       const int width, const ArrayTypes&... arrays )
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;

#if defined( CABANA_ENABLE_MPI_ADVANCE )
    return std::make_shared<
        StreamHalo<ExecutionSpace, memory_space, CommSpace::MpiAdvance>>(
        exec_space, pattern, width, arrays... );
#elif defined( CABANA_ENABLE_MPICH )
    return std::make_shared<
        StreamHalo<ExecutionSpace, memory_space, CommSpace::Mpich>>(
        exec_space, pattern, width, arrays... );
#elif defined( CABANA_ENABLE_CRAY_MPI )
    return std::make_shared<
        StreamHalo<ExecutionSpace, memory_space, CommSpace::CrayMpi>>(
        exec_space, pattern, width, arrays... );
#else
    return std::make_shared<
        StreamHalo<ExecutionSpace, memory_space, CommSpace::Mpi>>(
        exec_space, pattern, width, arrays... );
#endif
}

} // namespace Experimental
} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_STREAMHALO_HPP
