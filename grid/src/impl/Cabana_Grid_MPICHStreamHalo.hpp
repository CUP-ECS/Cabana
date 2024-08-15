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
namespace Impl
{

#ifdef Cabana_ENABLE_MPISTREAM_MPICH

// Backend for stream triggered halos using MPICH MPIX_Stream primitive

template <class MemorySpace>
class MPICHStreamHaloRequest
{
    using view_type = Kokkos::View<char*, MemorySpace>;

  public:
    MPICHStreamHaloRequest( const MPI_Comm &comm, const std::vector<int>& ranks,
                          const std::vector<view_type>& sendviews, 
                          const std::vector<view_type>& receiveviews )
        : _comm(comm), _ranks(ranks), _sendviews(sendviews), _receiveviews(receiveviews),
	  _requests(receiveviews.size(), MPI_REQUEST_NULL)
    {
    }
    void enqueueSend( int n )
    {
        if ( _sendviews[n].size() <= 0 )
	    return;
	
        MPIX_Send_enqueue( _sendviews[n].data(), _sendviews[n].size(),
                          MPI_BYTE, _ranks[n], 1234, _comm ); // XXX Get a real tag
    }
    void enqueueRecv( int n )
    {
        if (  _receiveviews[n].size() <= 0 ) {
            _requests[n] = MPI_REQUEST_NULL;
	    return;
        }
	
        MPIX_Irecv_enqueue( _receiveviews[n].data(), _receiveviews[n].size(),
                           MPI_BYTE, _ranks[n], 1234, _comm, &_requests[n] );

    }
    void enqueueSendAll( )
    {
        for (int i = 0; i < _sendviews.size(); i++) {
	    enqueueSend( i );
	}  
    }
    void enqueueRecvAll( )
    {
        for (int i = 0; i < _receiveviews.size(); i++) {
	    enqueueRecv( i );
	}  
    }
    void enqueueWaitAll( )
    {
        MPIX_Waitall_enqueue( _receiveviews.size(), _requests.data(), MPI_STATUSES_IGNORE );
    }

  private:
    const MPI_Comm &_comm;
    const std::vector<int>& _ranks; 
    const std::vector<view_type>& _sendviews;
    const std::vector<view_type>& _receiveviews;
    std::vector<MPI_Request> _requests;
};

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
                            std::vector<view_type> & sendviews,
			    std::vector<view_type> & receiveviews)
    {
        return std::make_unique<request_type>( _comm, Halo<MemorySpace>::_neighbor_ranks,
					       sendviews, receiveviews );
    }


    // Generic stream creation function for when we don't know how to get the underlying
    // device stream
    template <typename ExecSpace> 
    void
    createMPIXStream(const ExecSpace & exec_space)
        requires (std::same_as<ExecSpace, Kokkos::Serial>
#ifdef KOKKOS_ENABLE_OPENMP
                  || std::same_as<ExecSpace, Kokkos::OpenMP> 
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
                  || std::same_as<ExecSpace, Kokkos::OpenMPTarget> 
#endif
    //              || std::same_as<ExecSpace, Kokkos::DefaultExecutionSpace> 
        ) // end requires
    {
        MPIX_Stream_create(MPI_INFO_NULL, &_stream);
    }

#ifdef KOKKOS_ENABLE_CUDA   
    // Functions to create the stream communicator specialized for ones we know how to
    // access the device stream
    template <typename ExecSpace>
    void
    createMPIXStream(const ExecSpace & exec_space)
        requires (std::same_as<ExecSpace, Kokkos::Cuda>) 
    {
        MPI_Info i;
	cudaStream_t stream = exec_space.cuda_stream();
        MPI_Info_create(&i);
	MPI_Info_set(i, "type", "cudaStream_t");
        MPIX_Info_set_hex(i, "value", &stream, sizeof(stream));
        MPIX_Stream_create(i, &_stream);
        MPI_Info_free(&i);
    }
#endif
   
    template <class ExecSpace, class Pattern, class... ArrayTypes>
    MPICHStreamHalo( const ExecSpace &exec_space, 
                     const Pattern& pattern, const int width, const ArrayTypes&... arrays )
       : StreamHalo<MemorySpace, MPICHStreamHalo<MemorySpace>>(pattern, width, arrays...)
    {
        // Initialize the MPI_Stream from the exec_space
        createMPIXStream(exec_space);
        // Create the stream communicator MPICH uses
        MPIX_Stream_comm_create(Halo<MemorySpace>::getComm(arrays...), _stream, &_comm);
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

} // Impl
} // Grid
} // Cabana

#endif CABANA_GRID_MPICHSTREAMHALO_HPP
