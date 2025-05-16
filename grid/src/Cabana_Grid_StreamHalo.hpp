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

#include <Cabana_Types.hpp>
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
#include <iostream>
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
class StreamHaloBase
  : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using halo_type = Cabana::Grid::Halo<memory_space>;

  protected:
    template <class Pattern, class... ArrayTypes>
    StreamHaloBase( const ExecutionSpace& exec_space, const Pattern& pattern, 
                    const int width, const ArrayTypes&... arrays )
        : Cabana::Grid::Halo<MemorySpace>(pattern, width, arrays...),
          _exec_space(exec_space)
    {
    }

    //! Enqueue operations to pack arrays into a buffer. Calling code must fence.
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

    //! Enqueue operations to pack all arrays into a buffer. Calling code must fence.
    //! Separate from the code to pack a single buffer to enable a later version with
    //! a fused packing kernel
    template <class... ArrayViews>
    void enqueuePackBuffers( const std::vector<Kokkos::View<char*, memory_space>>& buffers,
                             const std::vector<Kokkos::View<int**, memory_space>>& steerings,
                             ArrayViews... array_views ) const
    {
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for (int n = 0; n < buffers.size(); n++) {
            Kokkos::View<char *, memory_space> buffer = buffers[n];
            Kokkos::View<int **, memory_space> steering = steerings[n];
            if (buffers[n].size() > 0) 
                enqueuePackBuffer(buffer, steering, array_views...);
        }
    }

    //! Enqueue operations to unpack buffer into arrays. Calling code must fence.
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

    //! Enqueue operations to unpack buffer into arrays. Calling code must fence.
    //! Separate from the code to unpack a single buffer to enable a later version with
    //! a fused unpacking kernel
    template <class ReduceOp, class... ArrayViews>
    void enqueueUnpackBuffers( const ReduceOp& reduce_op,
                              const std::vector<Kokkos::View<char*, memory_space>>& buffers,
                              const std::vector<Kokkos::View<int**, memory_space>>& steerings,
                              ArrayViews... array_views ) const
    {
        /* Figure out how to refactor this to fuse these kernel launches. Note
         * that the steering vectors can be of different lengths! XXX */
        for (int n = 0; n < buffers.size(); n++) {
            Kokkos::View<char *, memory_space> buffer = buffers[n];
            Kokkos::View<int **, memory_space> steering = steerings[n];
            if (buffers[n].size() > 0) 
                enqueueUnpackBuffer(reduce_op, buffer, steering, array_views...);
        }
    }
 
  protected:
    const execution_space _exec_space; // not a reference - we want the copy here.
};

template <class ExecutionSpace, class MemorySpace, class CommSpace = Cabana::CommSpace::MPI>
class StreamHalo;

template <class ExecutionSpace, class MemorySpace>
class StreamHalo<ExecutionSpace, MemorySpace, Cabana::CommSpace::MPI>
  : public StreamHaloBase<ExecutionSpace, MemorySpace>
{
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using halo_type = Cabana::Grid::Halo<memory_space>;
    using base_type = StreamHaloBase<execution_space, memory_space>;

  public:
    /*!
      \brief Vanilla MPI Stream-triggered version to gather data into our ghosts 
      from their owners. Note that this fences to emulate stream semantics.

      \param exec_space The execution space to use for pack/unpack.

      \param arrays The arrays to gather. NOTE: These arrays must be given in
      the same order as in the constructor. These could technically be
      different arrays, they just need to have the same layouts and data types
      as the input arrays.
    */
    template <class... ArrayTypes>
    void enqueueGather( const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo<Commspace::MPI>::gather" );
        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream 
        // is already using our buffers.
        this->_exec_space.fence();
        for (int n = 0; n < num_n; n++) {
            if ( halo_type::_ghosted_buffers[n].size()  <= 0 ) {
                _requests[num_n + n] = MPI_REQUEST_NULL;
            } else {
                MPI_Irecv( halo_type::_ghosted_buffers[n].data(), 
                           halo_type::_ghosted_buffers[n].size(),
                           MPI_BYTE, halo_type::_neighbor_ranks[n], 
                           1234 + halo_type::_receive_tags[n], _comm, 
                           &_requests[num_n + n] );
            }
        }

        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_owned_buffers, 
                                  halo_type::_owned_steering, arrays.view()... );
        this->_exec_space.fence(); 

        for (int n = 0; n < num_n; n++) {
            if ( halo_type::_owned_buffers[n].size() <= 0 ) {
                _requests[n] = MPI_REQUEST_NULL;
            } else {
                MPI_Isend( halo_type::_owned_buffers[n].data(), 
                           halo_type::_owned_buffers[n].size(), 
                           MPI_BYTE, halo_type::_neighbor_ranks[n], 
                           1234 + halo_type::_send_tags[n], _comm, 
                           &_requests[n] ); // XXX Get a real tag
            }
        }
        MPI_Waitall( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );

        this->enqueueUnpackBuffers( ScatterReduce::Replace(),
                                    halo_type::_ghosted_buffers, 
                                    halo_type::_ghosted_steering, 
                                    arrays.view()... );
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param reduce_op The functor used to reduce the results.
      \param exec_space The execution space to use for pack/unpack.
      \param arrays The arrays to scatter.
    */
    template <class ReduceOp, class... ArrayTypes>
    void enqueueScatter( const ReduceOp& reduce_op, const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo<Commspace::MPI>::scatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream 
        // is already using our buffers.
        this->_exec_space.fence();
        for (int n = 0; n < num_n; n++) {
            if ( halo_type::_owned_buffers[n].size()  <= 0 ) {
                _requests[num_n + n] = MPI_REQUEST_NULL;
               continue;
            }
            MPI_Irecv( halo_type::_owned_buffers[n].data(), 
                       halo_type::_owned_buffers[n].size(),
                       MPI_BYTE, halo_type::_neighbor_ranks[n], 
                       1234 + halo_type::_receive_tags[n], _comm, 
                       &_requests[num_n + n] );
        }

        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_ghosted_buffers, 
                                  halo_type::_ghosted_steering, arrays.view()... );
        this->_exec_space.fence(); 

        for (int n = 0; n < num_n; n++) {
            if ( halo_type::_ghosted_buffers[n].size() <= 0 ) {
                _requests[n] = MPI_REQUEST_NULL;
               continue;
            }
            MPI_Isend( halo_type::_ghosted_buffers[n].data(), 
                       halo_type::_ghosted_buffers[n].size(), 
                       MPI_BYTE, halo_type::_neighbor_ranks[n], 
                       1234 + halo_type::_send_tags[n], _comm, 
                       &_requests[n] ); 
        }

        MPI_Waitall( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );

        this->enqueueUnpackBuffers( reduce_op, halo_type::_owned_buffers, 
                                    halo_type::_owned_steering, 
                                    arrays.view()... );
    }

    template <class Pattern, class... ArrayTypes>
    StreamHalo( const ExecutionSpace &exec_space, const Pattern& pattern,
                       const int width, const ArrayTypes&... arrays )
       : StreamHaloBase<ExecutionSpace, MemorySpace>(exec_space, pattern, width, arrays...),
         _requests(2 * halo_type::_neighbor_ranks.size(), MPI_REQUEST_NULL),
         _comm(Halo<MemorySpace>::getComm(arrays...))
    {
    }

  private:
    const MPI_Comm _comm;
    std::vector<MPI_Request> _requests;
}; // StreamHalo<Commspace::MPI>

/*!
  \brief Halo creation function.
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param arrays The arrays over which to build the halo.
  \return Shared pointer to a Halo.
*/
template <class ExecutionSpace, class Pattern, class... ArrayTypes>
auto createStreamHalo( const ExecutionSpace& exec_space, 
                       const Pattern& pattern, const int width, 
                       const ArrayTypes&... arrays )
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;

#if defined(CABANA_ENABLE_MPIADVANCE)
    return std::make_shared<StreamHalo<ExecutionSpace, memory_space, CommSpace::MPIAdvance>>(exec_space, pattern, width, arrays...);
#elif defined(CABANA_ENABLE_MPICH)
    return std::make_shared<StreamHalo<ExecutionSpace, memory_space, CommSpace::MPICH>>(exec_space, pattern, width, arrays...);
#elif defined(CABANA_ENABLE_CRAYMPI)
    return std::make_shared<StreamHalo<ExecutionSpace, memory_space, CommSpace::CrayMPI>>(exec_space, pattern, width, arrays...);
#else
    return std::make_shared<StreamHalo<ExecutionSpace, memory_space, CommSpace::MPI>>(exec_space, pattern, width, arrays...);
#endif
}
} // namespace Experimental
} // namespace Grid
} // namespace Cabana

// Include further specializations of StreamHalo if enabled.

#ifdef Cabana_ENABLE_MPICH
#include <impl/Cabana_Grid_MPICHStreamHalo.hpp>
#endif // MPICH
#ifdef Cabana_ENABLE_CRAYPICH
#include <impl/Cabana_Grid_HPEStreamHalo.hpp>
#endif // HPE
#ifdef Cabana_ENABLE_MPIADVANCE
#include <impl/Cabana_Grid_MPIAdvanceStreamHalo.hpp>
#endif // MPIADVANCE

#endif // end CABANA_GRID_STREAMHALO_HPP
