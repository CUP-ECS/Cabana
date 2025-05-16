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
  \file Cabana_Grid_MPICHStreamHalo.hpp
  \brief Implementations of multi-node grid stream-based scatter/gather
*/
#ifndef CABANA_GRID_MPICHSTREAMHALO_HPP
#define CABANA_GRID_MPICHSTREAMHALO_HPP

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
namespace Experimental
{

#ifdef Cabana_ENABLE_MPICH

template <class ExecutionSpace, class MemorySpace>
class MPICHStreamHalo<class ExecutionSpace, class MemorySpace,
                      Cabana::ComSpace::MPICH>
    : public StreamHaloBase<ExecutionSpace, MemorySpace>
{
  public:
    using view_type = Kokkos::View<char*, MemorySpace>;
    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    /*!
      \brief Vanilla MPI Stream-triggered version to gather data into our
      ghosts from their owners. Note that this fences to emulate stream
      semantics.

      \param exec_space The execution space to use for pack/unpack.

      \param arrays The arrays to gather. NOTE: These arrays must be 
      given in the same order as in the constructor. These could technically 
      be different arrays, they just need to have the same layouts and data 
      types as the input arrays.
    */
    template <class... ArrayTypes>
    void enqueueGather( const ArrayTypes&... arrays )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::StreamHalo<Commspace::MPI>::gather" );
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
                _requests[n] = MPI_REQUEST_NULL;
            }
            else
            {
                if ( halo_type::_ghosted_buffers[n].size() > 0 )
                {
                    MPIX_Irecv_enqueue( halo_type::_ghosted_buffers[n].data(),
                                        halo_type::_ghosted_buffers[n].size(),
                                        MPI_BYTE, halo_type::_neighbor_ranks[n],
                                        1234 + halo_type::_receive_tags[n],
                                        _comm, &_requests[n] );
                }
            }
            // Pack and send the data.
            this->enqueuePackBuffers( halo_type::_owned_buffers,
                                      halo_type::_owned_steering,
                                      arrays.view()... );
            for ( int n = 0; n < num_n; n++ )
            {
                if ( halo_type::_owned_buffers[n].size() > 0 )
                    MPIX_Send_enqueue( halo_type::_owned_buffers[n].data(),
                                       halo_type::_owned_buffers[n].size(),
                                       MPI_BYTE, halo_type::_neighbor_ranks[n],
                                       1234 + halo_type::_send_tags[n], _comm );
            }
        }
        MPIX_Waitall_enqueue( _requests.size(), _requests.data(),
                              MPI_STATUSES_IGNORE );

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
            "Cabana::Grid::StreamHalo<Commspace::MPI>::scatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream
        // is already using our buffers.
        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_owned_buffers[n].size() <= 0 )
            {
                _requests[n] = MPI_REQUEST_NULL;
                continue;
            }
            MPI_Irecv_enqueue( halo_type::_owned_buffers[n].data(),
                               halo_type::_owned_buffers[n].size(), MPI_BYTE,
                               halo_type::_neighbor_ranks[n],
                               1234 + halo_type::_receive_tags[n], _comm,
                               &_requests[n] );
        }

        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_ghosted_buffers,
                                  halo_type::_ghosted_steering,
                                  arrays.view()... );

        for ( int n = 0; n < num_n; n++ )
        {
            if ( halo_type::_ghosted_buffers[n].size() > 0 )
            {
                MPI_Send_enqueue( halo_type::_ghosted_buffers[n].data(),
                                  halo_type::_ghosted_buffers[n].size(),
                                  MPI_BYTE, halo_type::_neighbor_ranks[n],
                                  1234 + halo_type::_send_tags[n], _comm );
            }
        }
        MPI_Waitall( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );

        this->enqueueUnpackBuffers( reduce_op, halo_type::_owned_buffers,
                                    halo_type::_owned_steering,
                                    arrays.view()... );
    }

    // Generic stream creation function for when we don't know how to get the
    // underlying device stream
    template <typename ExecSpace>
    void createMPIXStream( const ExecSpace& exec_space )
        requires( std::same_as<ExecSpace, Kokkos::Serial>
#ifdef KOKKOS_ENABLE_OPENMP
                  || std::same_as<ExecSpace, Kokkos::OpenMP>
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
                  || std::same_as<ExecSpace, Kokkos::OpenMPTarget>
#endif
                  ) // end requires
    {
        MPIX_Stream_create( MPI_INFO_NULL, &_stream );
    }

#ifdef KOKKOS_ENABLE_CUDA
    // Functions to create the stream communicator specialized for ones we know
    // how to access the device stream
    template <typename ExecSpace>
    void createMPIXStream( const ExecSpace& exec_space )
        requires( std::same_as<ExecSpace, Kokkos::Cuda> )
    {
        MPI_Info i;
        cudaStream_t stream = exec_space.cuda_stream();
        MPI_Info_create( &i );
        MPI_Info_set( i, "type", "cudaStream_t" );
        MPIX_Info_set_hex( i, "value", &stream, sizeof( stream ) );
        MPIX_Stream_create( i, &_stream );
        MPI_Info_free( &i );
    }
#endif

  public:
    template <class Pattern, class... ArrayTypes>
    MPICHStreamHalo( const ExecutionSpace& exec_space, const Pattern& pattern,
                     const int width, const ArrayTypes&... arrays )
        : StreamHaloBase<ExecutionSpace, MemorySpace>( exec_space, pattern,
                                                       width, arrays... )
        , _requests( halo_type::_neighbor_ranks.size(), MPI_REQUEST_NULL )
    {
        // Initialize the MPI_Stream from the exec_space
        createMPIXStream( exec_space );
        // Create the stream communicator MPICH uses
        MPIX_Stream_comm_create( halo_type::getComm( arrays... ), _stream,
                                 &_comm );
    }

    ~MPICHStreamHalo()
    {
        MPI_Comm_free(
            &_comm ); // XXX We need to worry about copy constructoin here. That
                      // should dup the comm and the stream?
        MPIX_Stream_free( &_stream );
    }

  private:
    MPIX_Stream _stream;
    MPI_Comm _comm;
    std::vector<MPI_Request> _requests;
};

template <class ExecSpace, class Pattern, class... ArrayTypes>
auto createMPICHStreamHalo( const ExecSpace& exec_space, const Pattern& pattern,
                            const int width, const ArrayTypes&... arrays )
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<MPICHStreamHalo<memory_space>>( exec_space, pattern,
                                                            width, arrays... );
}

} // Impl
} // Grid
} // Cabana

#endif CABANA_GRID_MPICHSTREAMHALO_HPP
