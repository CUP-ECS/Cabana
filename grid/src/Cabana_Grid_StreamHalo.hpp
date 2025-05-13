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

#include <Cabana_ParameterPack.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <Cuda/Kokkos_Cuda.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
// StreamHalo
// ---------------------------------------------------------------------------//
/*!
  Stream-triggered multiple array halo communication plan for migrating 
  shared data between blocks. The Derived template class is the subclass implementing the
  stream halo for a given stream triggered communication backend.
*/
template <class MemorySpace>
class StreamHalo
  : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    using memory_space = MemorySpace;
    using super_type = Cabana::Grid::Halo<memory_space>;

    template <class Pattern, class... ArrayTypes>
    StreamHalo( const Pattern& pattern, const int width, const ArrayTypes&... arrays )
        : Cabana::Grid::Halo<MemorySpace>(pattern, width, arrays...)
    {
    }

    /*!
      \brief Stream-triggered version to gather data into our ghosts from their owners.

      \param exec_space The execution space to use for pack/unpack.

      \param arrays The arrays to gather. NOTE: These arrays must be given in
      the same order as in the constructor. These could technically be
      different arrays, they just need to have the same layouts and data types
      as the input arrays.
    */
    template <class ExecutionSpace, class... ArrayTypes>
    void enqueueGather( const ExecutionSpace& exec_space,
                        const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo::enqueueGather" );

        // Get the number of neighbors. Return if we have none.
        int num_n = super_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

	// Start receives
        enqueueRecvAll( super_type::_ghosted_buffers );
        // Pack and send the data.
        enqueuePackBuffers( exec_space, super_type::_owned_buffers, 
                           super_type::_owned_steering, arrays.view()... );
        enqueueSendAll( super_type::_owned_buffers ); 

	// Wait for and unpack all received data
        enqueueWaitAll( );
        enqueueUnpackBuffers( ScatterReduce::Replace(), exec_space,
                              super_type::_ghosted_buffers, 
                              super_type::_ghosted_steering, arrays.view()... );
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param reduce_op The functor used to reduce the results.
      \param exec_space The execution space to use for pack/unpack.
      \param arrays The arrays to scatter.
    */
    template <class ExecutionSpace, class ReduceOp, class... ArrayTypes>
    void enqueueScatter( const ExecutionSpace& exec_space, const ReduceOp& reduce_op,
                  const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo::enqueueScatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = super_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Initialize sends and receives, either point to point or collective
        enqueueRecvAll( super_type::_owned_buffers);

        // Pack and send ghost data back to owner. 
        enqueuePackBuffers( exec_space, super_type::_ghosted_buffers, 
                                   super_type::_ghosted_steering, arrays.view()... );
        enqueueSendAll(super_type::_ghosted_buffers); 

	// Wait for and unpack all received data into the owned buffers
        enqueueWaitAll( );
        enqueueUnpackBuffers( reduce_op, exec_space, super_type::_owned_buffers, 
                              super_type::_owned_steering, arrays.view()... );
    }

  public:
    //! Enqueue operations to pack arrays into a buffer. Calling code must fence.
    template <class ExecutionSpace, class... ArrayViews>
    void enqueuePackBuffers( const ExecutionSpace& exec_space,
                             const std::vector<Kokkos::View<char*, memory_space>>& buffers,
                             const std::vector<Kokkos::View<int**, memory_space>>& steerings,
                             ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for (int n = 0; n < buffers.size(); n++) {
            Kokkos::View<char *, memory_space> buffer = buffers[n];
            Kokkos::View<char *, memory_space> steering = steerings[n];
            Kokkos::parallel_for(
                "Cabana::Grid::StreamHalo::pack_buffer",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                     steering[n].extent( 0 ) ),
                KOKKOS_LAMBDA( const int i ) {
                    super_type::packArray(
                        buffer, steering, i,
                        std::integral_constant<std::size_t,
                                               sizeof...( ArrayViews ) - 1>(),
                        pp );
                } );
        }
    }

    //! Enqueue operations to unpack buffer into arrays. Calling code must fence.
    template <class ExecutionSpace, class ReduceOp, class... ArrayViews>
    void enqueueUnpackBuffers( const ReduceOp& reduce_op,
                              const ExecutionSpace& exec_space,
                              const std::vector<Kokkos::View<char*, memory_space>>& buffers,
                              const std::vector<Kokkos::View<int**, memory_space>>& steerings,
                              ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for (int n = 0; n < buffers.size(); n++) {
            Kokkos::View<char *, memory_space> buffer = buffers[n];
            Kokkos::View<char *, memory_space> steering = steerings[n];
            Kokkos::parallel_for(
                "Cabana::Grid::StreamHalo::unpack_buffer",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                     steering.extent( 0 ) ),
                KOKKOS_LAMBDA( const int i ) {
                    super_type::unpackArray(
                        reduce_op, buffer, steering, i,
                        std::integral_constant<std::size_t,
                                               sizeof...( ArrayViews ) - 1>(),
                        pp );
                } );
        }
    }
  protected:
    virtual void enqueueSendAll( std::vector<Kokkos::View<char*, MemorySpace>> & sendviews) = 0;
    virtual void enqueueRecvAll( std::vector<Kokkos::View<char*, MemorySpace>> & recvviews) = 0;
    virtual void enqueueWaitall() = 0;
};

} // namespace Grid
} // namespace Cabana

#ifdef Cabana_ENABLE_MPI
#include <impl/Cabana_Grid_VanillaStreamHalo.hpp>
#endif // MPI
#ifdef Cabana_ENABLE_MPICH
#include <impl/Cabana_Grid_MPICHStreamHalo.hpp>
#endif // MPICH
#ifdef Cabana_ENABLE_HPE
#include <impl/Cabana_Grid_HPEStreamHalo.hpp>
#endif // HPE
#ifdef Cabana_ENABLE_MPIADVANCE
#include <impl/Cabana_Grid_MPIAdvanceStreamHalo.hpp>
#endif // MPIADVANCE

#endif // end CABANA_GRID_STREAMHALO_HPP
