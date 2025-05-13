
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
  \file Cabana_Grid_VanillaStreamHalo.hpp
  \brief Implementations of multi-node grid stream-based scatter/gather
*/
#ifndef CABANA_GRID_VANILLASTREAMHALO_HPP
#define CABANA_GRID_VANILLASTREAMHALO_HPP

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

#ifdef Cabana_ENABLE_MPISTREAM_VANILLA

// Backend for emulating stream-triggered communication using vanilla MPI
template <class MemorySpace>
class VanillaStreamHaloRequest
{
    using view_type = Kokkos::View<char*, MemorySpace>;

  public:
    VanillaStreamHaloRequest( const MPI_Comm &comm, const std::vector<int> ranks,
                              const std::vector<view_type> sendviews, 
                              const std::vector<view_type> receiveviews )
        : _comm(comm), _ranks(ranks), _sendviews(sendviews), _receiveviews(receiveviews),
	  _requests(sendviews.size() + receiveviews.size(), MPI_REQUEST_NULL)
    {
    
    }


template <class MemorySpace>
class VanillaStreamHalo
   : public StreamHalo<MemorySpace, VanillaStreamHalo<MemorySpace>>
{
  public:
    using view_type = Kokkos::View<char*, MemorySpace>;
    using request_type = VanillaStreamHaloRequest<MemorySpace>;

  protected:
    virtual void enqueueSendAll( std::vector<Kokkos::View<char*, MemorySpace>> & sendviews) override
    {
        Kokkos::fence(); // XXX Should be able to pass in an execution space instance
        for (int n = 0; n < _sendviews.size(); n++) {
            if ( sendviews[n].size() <= 0 ) {
                _requests[n] = MPI_REQUEST_NULL;
	        continue; 
            }
            MPI_Isend( sendviews[n].data(), sendviews[n].size(),
                       MPI_BYTE, halo_type::_neighbor_ranks[n], 1234, _comm, &_requests[n] ); // XXX Get a real tag
	}  
    }
    virtual void enqueueRecvAll( std::vector<Kokkos::View<char*, MemorySpace>> & recvviews) override
    {
        int nsends = _sendviews.size();
        Kokkos::fence(); // XXX Should be able to pass in an execution space instance
        for (int n = 0; n < _receiveviews.size(); n++) {
            if ( _receiveviews[n].size() <= 0 ) {
                _requests[nsends + n] = MPI_REQUEST_NULL;
	       continue;
            }
            MPI_Irecv( _receiveviews[n].data(), _receiveviews[n].size(),
                       MPI_BYTE, halo_type::_neighbor_ranks[n], 1234, _comm, &_requests[nsends + n] );
	}  
    }
    void enqueueWaitAll( )
    {
        Kokkos::fence();
        MPI_Waitall( _sendviews.size() + _receiveviews.size(), 
                     _requests.data(), MPI_STATUSES_IGNORE );
    }

    template <class ExecSpace, class Pattern, class... ArrayTypes>
    VanillaStreamHalo( const ExecSpace &exec_space, const Pattern& pattern, 
                       const int width, const ArrayTypes&... arrays )
       : StreamHalo<MemorySpace, VanillaStreamHalo<MemorySpace>>(pattern, width, arrays...),
         _requests(2 * halo_type::_neighbor_ranks.size(), MPI_REQUEST_NULL)
    {
        _comm = Halo<MemorySpace>::getComm(arrays...);
    }

  private:
    const MPI_Comm _comm;
    std::vector<MPI_Request> _requests;
};

template <class ExecSpace, class Pattern, class... ArrayTypes>
auto createVanillaStreamHalo( const ExecSpace & exec_space, const Pattern &pattern, 
                              const int width, const ArrayTypes&... arrays)
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<VanillaStreamHalo<memory_space>>( exec_space, 
        pattern, width, arrays... );
}
#endif // Cabana_ENABLE_MPISTREAM_VANILLA
} // Impl
} // Grid
} // Cabana

#endif CABANA_GRID_VANILLASTREAMHALO_HPP
