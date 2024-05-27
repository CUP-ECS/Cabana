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
template <class MemorySpace, class Derived>
class StreamHalo
  : public Cabana::Grid::Halo<MemorySpace>
{
  public:
    using memory_space = MemorySpace;
    using backend_type = Derived;
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
                        const ArrayTypes&... arrays ) const
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo::enqueueGather" );

        // Get the number of neighbors. Return if we have none.
        int num_n = super_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Initialize sends and receives, either point to point or collective
        auto r = static_cast<Derived *>(this)->createBackendRequest(exec_space, 
            super_type::_neighbor_ranks, super_type::_owned_buffers, 
            super_type::_ghosted_buffers );

        r->enqueueRecvsReady( );

        // Pack and send owned data to ghosts
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < super_type::_ghosted_buffers[n].size() )
            {
                // Pack the send buffer.
                enqueuePackBuffer( exec_space, super_type::_owned_buffers[n], 
                                   super_type::_owned_steering[n], arrays.view()... );
                r->enqueueSendReady(n); 
            }
        }
     
        r->enqueueWaitall( );

        // Launch kernels to unpack receive buffers.
        for ( int n = 0; n < num_n; ++n )
        {
            if ( 0 < super_type::_ghosted_buffers[n].size() ) 
                enqueueUnpackBuffer( ScatterReduce::Replace(), exec_space,
                                     super_type::_ghosted_buffers[n], 
                                     super_type::_ghosted_steering[n], arrays.view()... );
        }
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
                  const ArrayTypes&... arrays ) const
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::StreamHalo::enqueueScatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = super_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Initialize sends and receives, either point to point or collective
        auto r = static_cast<Derived *>(this)->createBackendRequest(exec_space, 
            super_type::_neighbor_ranks, super_type::_owned_buffers, 
            super_type::_ghosted_buffers );
        r->enqueueRecvsReady( );

        // Pack and send ghost data back to owner. XXX Would be good to explore a single
        // fused pack kernel
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < super_type::_owned_buffers[n].size() )
            {
                // Pack the send buffer.
                enqueuePackBuffer( exec_space, super_type::_ghosted_buffers[n], 
                                   super_type::_ghosted_steering[n], arrays.view()... );
                r->enqueueSendReady(n); 
            }
        }
     
        r->enqueueWaitall( );

        // Launch kernels to unpack receive buffers. XXX We should make a single 
        // fused unpack kernel
        for ( int n = 0; n < num_n; ++n )
        {
            if ( 0 < super_type::_owned_buffers[n].size() ) 
                enqueueUnpackBuffer( reduce_op, exec_space,
                                     super_type::_owned_buffers[n], 
                                     super_type::_owned_steering[n], arrays.view()... );
        }
    }

  public:
    //! Enqueue operations to pack arrays into a buffer. Calling code must fence.
    template <class ExecutionSpace, class... ArrayViews>
    void enqueuePackBuffer( const ExecutionSpace& exec_space,
                            const Kokkos::View<char*, memory_space>& buffer,
                            const Kokkos::View<int**, memory_space>& steering,
                            ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::Halo::pack_buffer",
            Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                 steering.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                super_type::packArray(
                    buffer, steering, i,
                    std::integral_constant<std::size_t,
                                           sizeof...( ArrayViews ) - 1>(),
                    pp );
            } );
    }

    //! Enqueue operations to unpack buffer into arrays. Calling code must fence.
    template <class ExecutionSpace, class ReduceOp, class... ArrayViews>
    void enqueueUnpackBuffer( const ReduceOp& reduce_op,
                              const ExecutionSpace& exec_space,
                              const Kokkos::View<char*, memory_space>& buffer,
                              const Kokkos::View<int**, memory_space>& steering,
                              ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::Halo::unpack_buffer",
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
};

//---------------------------------------------------------------------------//

namespace Impl
{

#ifdef Cabana_MPISTREAM_HPE_ENABLED
   
template <class ExecSpace, class MemorySpace>
class HPEStreamHaloRequest
{
    using view_type = Kokkos::View<char*, MemorySpace>;

    HPEStreamHaloRequest( const MPI_Queue &queue, const std::vector<int> ranks,
                          const std::vector<view_type> sendviews, 
                          const std::vector<view_type> receiviews )
        : _queue(queue), _ranks(ranks), _sendviews(sendviews), _receiveviews(receiveviews)
    {
    }
  public:
    void enqueueSend( int n )
    {
    }
    void enqueueRecv( int n )
    {
    }
    void enqueueWait( int n )
    {
    }
    void enqueueSendAll( )
    {
    }
    void enqueueRecvAll( )
    {
    }
    void enqueueWaitAll( )
    {
    }

  private:
    const MPI_Queue &_queue;
    const std::vector<int>& _ranks; 
    const std::vector<view_type>& _sendviews, _receiveviews; 
};

// Backend for one-sided stream-based communication using 
template <class MemorySpace>
class HPEStreamHalo
   : public StreamHalo<MemorySpace, HPEStreamHalo>
{
    using view_type = Kokkos::View<char*, MemorySpace>;
    using request_type = HPEStreamHaloRequest<MemorySpace>;

    template <class ExecSpace>
    std::unique_ptr<request_type> 
    createStreamHaloRequest(const ExecSpace & exec_space, 
                            view_type sendviews, view_type recvviews)
    {
        return std::make_unique<request_type>( queue, Halo::_ranks, sendviews, recvviews );l
    }

    template <class ExecSpace, class Pattern, class ArrayTypes...>
    HPEStreamHalo( const ExecSpace &exec_space, 
                   const Pattern& pattern, const int width, const ArrayTypes&... arrays )
    {
        // Initialize the MPI_Queue from the exec_space
    }
  private:
    const MPI_Queue _queue;
};

template <class ExecSpace, class Pattern, class... ArrayTypes>
auto createHPEStreamHalo( const ExecSpace & exec_space, 
                          const Pattern &pattern, const int width, const ArrayTypes&... arrays)
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<HPEStreamHalo<memory_space>>( exec_space, 
        pattern, width, arrays... );
}
#endif // Cabana_MPISTREAM_HPE_ENABLED

// #ifdef Cabana_MPISTREAM_MPICH_ENABLED
   
template <class MemorySpace>
class MPICHStreamHaloRequest
{
    using view_type = Kokkos::View<char*, MemorySpace>;

    MPICHStreamHaloRequest( const MPI_Comm &comm, const std::vector<int> ranks,
                          const std::vector<view_type> sendviews, 
                          const std::vector<view_type> receiveviews )
        : _comm(comm), _ranks(ranks), _sendviews(sendviews), _receiveviews(receiveviews)
    {
    }
  public:
    void enqueueSend( int n )
    {
    }
    void enqueueRecv( int n )
    {
    }
    void enqueueWait( int n )
    {
    }
    void enqueueSendAll( )
    {
    }
    void enqueueRecvAll( )
    {
    }
    void enqueueWaitAll( )
    {
    }

  private:
    const MPI_Comm &_comm;
    const std::vector<int>& _ranks; 
    const std::vector<view_type>& _sendviews, _receiveviews; 
};

// Backend for one-sided stream-based communication using 
template <class MemorySpace>
class MPICHStreamHalo
   : public StreamHalo<MemorySpace, MPICHStreamHalo<MemorySpace>>
{
  public:
    using view_type = Kokkos::View<char*, MemorySpace>;
    using request_type = MPICHStreamHaloRequest<MemorySpace>;

    template <class ExecSpace>
    std::unique_ptr<request_type> 
    createStreamHaloRequest(const ExecSpace & exec_space, 
                            view_type sendviews, view_type receiveviews)
    {
        return std::make_unique<request_type>( _comm, Halo<MemorySpace>::_ranks,
					       sendviews, receiveviews );
    }

    template <class ExecSpace, class Pattern, class... ArrayTypes>
    MPICHStreamHalo( const ExecSpace &exec_space, 
                     const Pattern& pattern, const int width, const ArrayTypes&... arrays )
       : StreamHalo<MemorySpace, MPICHStreamHalo<MemorySpace>>(pattern, width, arrays...)
    {
        // Initialize the MPI_Stream from the exec_space
        // Create the stream communicator MPICH uses
    }
  private:
    MPIX_Stream _stream;
    MPI_Comm _comm;

};

template <class ExecSpace, class Pattern, class... ArrayTypes>
auto createMPICHStreamHalo( const ExecSpace & exec_space, 
                          const Pattern &pattern, const int width, const ArrayTypes&... arrays)
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<MPICHStreamHalo<memory_space>>( exec_space, 
        pattern, width, arrays... );
}
// #endif Cabana_MPISTREAM_MPICH_ENABLED

} // namespace Impl

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_STREAMHALO_HPP
