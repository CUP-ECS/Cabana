/****************************************************************************
 * Copyright (c) 2018-2025 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_MPIStreamHalo.hpp
  \brief Multi-node grid stream-based scatter/gather using MPI
*/
#ifndef CABANA_GRID_MPISTREAMHALO_HPP
#define CABANA_GRID_MPISTREAMHALO_HPP

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

template <class ExecutionSpace, class MemorySpace>
class StreamHalo<ExecutionSpace, MemorySpace, Cabana::CommSpace::Mpi>
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
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::StreamHalo<Commspace::Mpi>::gather" );
        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream
        // is already using our buffers.
        this->_exec_space.fence();
        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_ghosted_buffers[n].size() <= 0 )
            {
                _requests[num_n + n] = MPI_REQUEST_NULL;
            }
            else
            {
                MPI_Irecv( halo_type::_ghosted_buffers[n].data(),
                           halo_type::_ghosted_buffers[n].size(), MPI_BYTE,
                           halo_type::_neighbor_ranks[n],
                           1234 + halo_type::_receive_tags[n], _comm,
                           &_requests[num_n + n] );
            }
        }

        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_owned_buffers,
                                  halo_type::_owned_steering,
                                  arrays.view()... );
        this->_exec_space.fence();

        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_owned_buffers[n].size() <= 0 )
            {
                _requests[n] = MPI_REQUEST_NULL;
            }
            else
            {
                MPI_Isend( halo_type::_owned_buffers[n].data(),
                           halo_type::_owned_buffers[n].size(), MPI_BYTE,
                           halo_type::_neighbor_ranks[n],
                           1234 + halo_type::_send_tags[n], _comm,
                           &_requests[n] ); // XXX Get a real tag
            }
        }
        MPI_Waitall( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );

        this->enqueueUnpackBuffers(
            ScatterReduce::Replace(), halo_type::_ghosted_buffers,
            halo_type::_ghosted_steering, arrays.view()... );
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param reduce_op The functor used to reduce the results.
      \param exec_space The execution space to use for pack/unpack.
      \param arrays The arrays to scatter.
    */
    template <class ReduceOp, class... ArrayTypes>
    void enqueueScatter( const ReduceOp& reduce_op,
                         const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::StreamHalo<Commspace::Mpi>::scatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream
        // is already using our buffers.
        this->_exec_space.fence();
        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_owned_buffers[n].size() <= 0 )
            {
                _requests[num_n + n] = MPI_REQUEST_NULL;
                continue;
            }
            MPI_Irecv( halo_type::_owned_buffers[n].data(),
                       halo_type::_owned_buffers[n].size(), MPI_BYTE,
                       halo_type::_neighbor_ranks[n],
                       1234 + halo_type::_receive_tags[n], _comm,
                       &_requests[num_n + n] );
        }

        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_ghosted_buffers,
                                  halo_type::_ghosted_steering,
                                  arrays.view()... );
        this->_exec_space.fence();

        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_ghosted_buffers[n].size() <= 0 )
            {
                _requests[n] = MPI_REQUEST_NULL;
                continue;
            }
            MPI_Isend( halo_type::_ghosted_buffers[n].data(),
                       halo_type::_ghosted_buffers[n].size(), MPI_BYTE,
                       halo_type::_neighbor_ranks[n],
                       1234 + halo_type::_send_tags[n], _comm, &_requests[n] );
        }

        MPI_Waitall( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );

        this->enqueueUnpackBuffers( reduce_op, halo_type::_owned_buffers,
                                    halo_type::_owned_steering,
                                    arrays.view()... );
    }

    template <class Pattern, class... ArrayTypes>
    StreamHalo( const ExecutionSpace& exec_space, const Pattern& pattern,
                const int width, const ArrayTypes&... arrays )
        : StreamHaloBase<ExecutionSpace, MemorySpace>( exec_space, pattern,
                                                       width, arrays... )
        , _requests( 2 * halo_type::_neighbor_ranks.size(), MPI_REQUEST_NULL )
        , _comm( Halo<MemorySpace>::getComm( arrays... ) )
    {
    }

  private:
    const MPI_Comm _comm;
    std::vector<MPI_Request> _requests;
}; // StreamHalo<Commspace::Mpi>

} // namespace Experimental
} // namespace Grid
} // namespace Cabana
#endif // CABANA_GRID_MPISTREAMHALO_HPP
