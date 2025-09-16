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
#ifndef CABANA_GRID_MPIADVANCESTREAMHALO_HPP
#define CABANA_GRID_MPIADVANCESTREAMHALO_HPP

#include <Cabana_Types.hpp>
#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_ParameterPack.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <Cuda/Kokkos_Cuda.hpp>
#include <hip/hip_runtime.h>
#include <stream-triggering.h>

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
class StreamHalo<ExecutionSpace, MemorySpace, Cabana::CommSpace::MpiAdvance>
    : public StreamHaloBase<ExecutionSpace, MemorySpace>
{
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using halo_type = Cabana::Grid::Halo<memory_space>;
    using base_type = StreamHaloBase<execution_space, memory_space>;

  public:

  void* setup(const ExecutionSpace& a){
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::HIP>){
      //std::cout << "HIP STREAM!" << std::endl; // debug?
      return a.hip_stream();
    }
    else
    {
      //std::cout << "Base" << std::endl;
      return nullptr;
    }
  }
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
            "Cabana::Grid::StreamHalo<Commspace::MpiAdvance>::gather" );
        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // We fence before posting receives in case the stream
        // is already using our buffers.
	MPIS_Enqueue_startall(my_queue, num_n, &gather_requests[0]);
        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_owned_buffers,
                                  halo_type::_owned_steering,
                                  arrays.view()... );

	// do sends which are the last half of gather_requests
	MPIS_Enqueue_startall(my_queue, num_n, &gather_requests[num_n]);
	MPIS_Enqueue_waitall(my_queue);
	
        // Unpack receive buffers.        
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
            "Cabana::Grid::StreamHalo<Commspace::MpiAdvance>::scatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = halo_type::_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

	// Begin Recv inits
	MPIS_Enqueue_startall(my_queue, num_n, &scatter_requests[0]);
        // Pack and send the data.
        this->enqueuePackBuffers( halo_type::_ghosted_buffers,
                                  halo_type::_ghosted_steering,
                                  arrays.view()... );
	
	// Begin Send inits
	MPIS_Enqueue_startall(my_queue, num_n, &scatter_requests[num_n]);
	
	MPIS_Enqueue_waitall(my_queue);

        this->enqueueUnpackBuffers( reduce_op, halo_type::_owned_buffers,
                                    halo_type::_owned_steering,
                                    arrays.view()... );
    }

    template <class Pattern, class... ArrayTypes>
    StreamHalo( const ExecutionSpace& exec_space, const Pattern& pattern,
                const int width, const ArrayTypes&... arrays )
        : StreamHaloBase<ExecutionSpace, MemorySpace>( exec_space, pattern,
                                                       width, arrays... )
        , scatter_requests( 2 * halo_type::_neighbor_ranks.size(), MPIS_REQUEST_NULL )
	, gather_requests( 2 * halo_type::_neighbor_ranks.size(), MPIS_REQUEST_NULL )
        , _comm( Halo<MemorySpace>::getComm( arrays... ) )
    {
      int num_n = halo_type::_neighbor_ranks.size();

      // set up queue stream, if hip stream returns executionspace.hip_stream
      // else use nullptr for default functionality
      my_stream = setup(exec_space);
      // Note, change for other systems
      MPIS_Queue_init(&my_queue, CXI, &my_stream);

      MPI_Info_create(&mem_info);
      // Note, change for other systems
      //MPI_Info_set(mem_info, "MPIS_GPU_MEM_TYPE", "COARSE");
      MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");

      // scatter
      for ( int n = 0; n < num_n; ++n )
      {
	// Only process this neighbor if there is work to do.
	if ( 0 < halo_type::_owned_buffers[n].size() )
          {
	    // scatter
            MPIS_Recv_init( halo_type::_owned_buffers[n].data(),
			    halo_type::_owned_buffers[n].size(), MPI_BYTE,
			    halo_type::_neighbor_ranks[n],
			    1234 + halo_type::_receive_tags[n], _comm,
			    MPI_INFO_NULL,
			    &scatter_requests[n] );
          }
	// Post a send.
	if ( 0 < halo_type::_ghosted_buffers[n].size() )
            {
	    MPIS_Send_init( halo_type::_ghosted_buffers[n].data(),
			    halo_type::_ghosted_buffers[n].size(), MPI_BYTE,
			    halo_type::_neighbor_ranks[n],
			    1234 + halo_type::_send_tags[n], _comm,
			    mem_info,
			    &scatter_requests[num_n + n] );
	    }
      }

    // gather
    for ( int n = 0; n < num_n; ++n )
      {
	if ( 0 < halo_type::_ghosted_buffers[n].size() )
	  {
	    // Only process this neighbor if there is work to do.
	    MPIS_Recv_init( halo_type::_ghosted_buffers[n].data(),
			    halo_type::_ghosted_buffers[n].size(), MPI_BYTE,
			    halo_type::_neighbor_ranks[n],
			    5678 + halo_type::_receive_tags[n], _comm,
			    MPI_INFO_NULL,
			    &gather_requests[n] );
      }
	if ( 0 < halo_type::_owned_buffers[n].size() )
          {
	    MPIS_Send_init( halo_type::_owned_buffers[n].data(),
			    halo_type::_owned_buffers[n].size(), MPI_BYTE,
			    halo_type::_neighbor_ranks[n],
			    5678 + halo_type::_send_tags[n], _comm,
			    mem_info,
			    &gather_requests[num_n + n] ); 
          }
      }

    // match requests = count of gather + count of scatter, 4*num_n        
    std::vector<MPIS_Request> match_requests(4*num_n, MPIS_REQUEST_NULL);

    //for(int n = 0; n < 2*num_n; ++n)
    for(int n = 0, m = 0; n < 2*num_n; ++n, m+=2)
      {
        MPIS_Imatch(&scatter_requests[n], &match_requests[m] );
        MPIS_Imatch(&gather_requests[n], &match_requests[m+1]);

      }
    MPIS_Waitall(4*num_n, &match_requests[0], MPI_STATUSES_IGNORE);
    
    }
  
  
  ~StreamHalo(){
    for( int n =0; n < 2*halo_type::_neighbor_ranks.size(); ++n)
      {
	MPIS_Request_free(&scatter_requests[n]);
	MPIS_Request_free(&gather_requests[n]);
      }
    MPIS_Queue_free(&my_queue);
    MPI_Info_free(&mem_info);
  }

  private:
  void* my_stream;
  const MPI_Comm _comm;
  MPIS_Queue my_queue;
  std::vector<MPIS_Request> scatter_requests;
  std::vector<MPIS_Request> gather_requests;
  MPI_Info mem_info;
  
}; // StreamHalo<Commspace::MpiAdvance>
  
} // namespace Experimental
} // namespace Grid
} // namespace Cabana
#endif // CABANA_GRID_MPIAdvanceSTREAMHALO_HPP
