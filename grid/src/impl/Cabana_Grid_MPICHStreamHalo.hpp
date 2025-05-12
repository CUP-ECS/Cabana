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

#ifdef Cabana_ENABLE_MPICH

template <class MemorySpace>
class MPICHStreamHalo
   : public StreamHalo<MemorySpace, MPICHStreamHalo<MemorySpace>>
{
  public:
    using view_type = Kokkos::View<char*, MemorySpace>;
    using halo_type = Cabana:Grid:Halo<MemorySpace>;

  protected:
    virtual void enqueueSend( Kokkos::View<char*, MemorySpace> sendview, int rank ) override
    {
        if ( sendview.size() <= 0 )
	    return;
	
        MPIX_Send_enqueue( sendview.data(), sendview.size(),
                          MPI_BYTE, rank, 1234, _comm ); // XXX Get a real tag
    }

    virtual void void enqueueRecv( Kokkos::View<char*, MemorySpace> receiveview, int rank ) override
    {
        if (  receiveviews <= 0 ) {
            _requests[n] = MPI_REQUEST_NULL;
	    return;
        }
	
        MPIX_Irecv_enqueue( receiveview.data(), receiveviews.size(),
                           MPI_BYTE, rank, 1234, _comm, &_requests[n] );

    }
    virtual void enqueueSendAll( std::vector<Kokkos::View<char*, MemorySpace>> & sendviews) override
    {
        for (int i = 0; i < sendviews.size(); i++) {
	    enqueueSend( sendviews[i], i );
	}  
    }
    virtual void enqueueRecvAll( std::vector<Kokkos::View<char*, MemorySpace>> & recvviews) override
    {
        for (int i = 0; i < receiveviews.size(); i++) {
	    enqueueRecv( recvviews[i], i );
	}  
    }
    virtual void enqueueWaitAll( ) override
    {
        MPIX_Waitall_enqueue( _requests.size(), _requests.data(), MPI_STATUSES_IGNORE );
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
   
  public:
    template <class ExecSpace, class Pattern, class... ArrayTypes>
    MPICHStreamHalo( const ExecSpace &exec_space, 
                     const Pattern& pattern, const int width, const ArrayTypes&... arrays )
       : StreamHalo<MemorySpace>(pattern, width, arrays...),
         _requests(halo_type::_neighbor_ranks.size(), MPI_REQUEST_NULL)
    {
        // Initialize the MPI_Stream from the exec_space
        createMPIXStream(exec_space);
        // Create the stream communicator MPICH uses
        MPIX_Stream_comm_create(halo_type::getComm(arrays...), _stream, &_comm);
    }

  private:
    MPIX_Stream _stream;
    MPI_Comm _comm;
    std::vector<MPI_Request> _requests;
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
