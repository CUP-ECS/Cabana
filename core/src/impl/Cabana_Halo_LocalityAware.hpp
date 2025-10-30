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
  \file Cabana_comm_plan_LocalityAware.hpp
  \brief Multi-node particle scatter/gather, LocalityAware implementations
*/
#ifndef CABANA_HALO_LOCALITYAWARE_HPP
#define CABANA_HALO_LOCALITYAWARE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_CommunicationPlanBase.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <exception>
#include <vector>

namespace Cabana
{

/*!
    \brief Perform the gather operation.
*/
template <class HaloType, class AoSoAType>
template <class ExecutionSpace, class CommSpaceType>
std::enable_if_t<std::is_same<CommSpaceType, LocalityAware>::value, void>
Gather<HaloType, AoSoAType,
       typename std::enable_if<is_aosoa<AoSoAType>::value>::type>::
    applyImpl( ExecutionSpace, CommSpaceType )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::gather" );

    // Setup persistent communication if not already done
    if (!this->persistent_set()) {
        this->computeSendRecvData();
    }

    // Get the buffers and particle data (local copies for lambdas below).
    auto send_buffer = this->getSendBuffer();
    auto recv_buffer = this->getReceiveBuffer();
    auto aosoa = this->getData();

    // Get the steering vector for the sends.
    auto steering = _comm_plan.getExportSteering();
    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        send_buffer( i ) = aosoa.getTuple( steering( i ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> send_policy( 0, _send_size );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer", send_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Communicate data
    MPI_Status status;
    MPIL_Start(this->lrequest());
    MPIL_Wait(this->lrequest(), &status);

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = _comm_plan.numLocal();
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        std::size_t ghost_idx = i + num_local;
        aosoa.setTuple( ghost_idx, recv_buffer( i ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> recv_policy( 0, _recv_size );
    Kokkos::parallel_for( "Cabana::gather::apply::extract_recv_buffer",
                          recv_policy, extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( _comm_plan.comm() );
}

/*!
    \brief Perform the gather operation.
*/
template <class HaloType, class SliceType>
template <class ExecutionSpace, class CommSpaceType>
std::enable_if_t<std::is_same<CommSpaceType, LocalityAware>::value, void>
Gather<HaloType, SliceType,
       typename std::enable_if<is_slice<SliceType>::value>::type>::
    applyImpl( ExecutionSpace, CommSpaceType )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::gather" );

    // Setup persistent communication if not already done
    if (!this->persistent_set()) {
        this->computeSendRecvData();
    }

    // Get the buffers (local copies for lambdas below).
    auto send_buffer = this->getSendBuffer();
    auto recv_buffer = this->getReceiveBuffer();
    auto slice = this->getData();

    // Get the number of components in the slice.
    std::size_t num_comp = this->getSliceComponents();

    // Get the raw slice data.
    auto slice_data = slice.data();

    // Get the steering vector for the sends.
    auto steering = _comm_plan.getExportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto s = SliceType::index_type::s( steering( i ) );
        auto a = SliceType::index_type::a( steering( i ) );
        std::size_t slice_offset = s * slice.stride( 0 ) + a;
        for ( std::size_t n = 0; n < num_comp; ++n )
            send_buffer( i, n ) =
                slice_data[slice_offset + n * SliceType::vector_length];
    };
    Kokkos::RangePolicy<ExecutionSpace> send_policy( 0, _send_size );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer", send_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Communicate data
    MPI_Status status;
    MPIL_Start(this->lrequest());
    MPIL_Wait(this->lrequest(), &status);

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = _comm_plan.numLocal();
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        std::size_t ghost_idx = i + num_local;
        auto s = SliceType::index_type::s( ghost_idx );
        auto a = SliceType::index_type::a( ghost_idx );
        std::size_t slice_offset = s * slice.stride( 0 ) + a;
        for ( std::size_t n = 0; n < num_comp; ++n )
            slice_data[slice_offset + SliceType::vector_length * n] =
                recv_buffer( i, n );
    };
    Kokkos::RangePolicy<ExecutionSpace> recv_policy( 0, _recv_size );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer", recv_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( _comm_plan.comm() );
}

/**********
 * SCATTER *
 **********/

template <class HaloType, class SliceType>
template <class ExecutionSpace, class CommSpaceType>
std::enable_if_t<std::is_same<CommSpaceType, LocalityAware>::value, void>
Scatter<HaloType, SliceType>::applyImpl( ExecutionSpace, CommSpaceType )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::scatter" );

    // Setup persistent communication if not already done
    if (!this->persistent_set()) {
        this->computeSendRecvData();
    }

    // Get the buffers (local copies for lambdas below).
    auto send_buffer = this->getSendBuffer();
    auto recv_buffer = this->getReceiveBuffer();
    auto slice = this->getData();

    // Get the number of components in the slice.
    std::size_t num_comp = this->getSliceComponents();

    // Get the raw slice data. Wrap in a 1D Kokkos View so we can unroll the
    // components of each slice element.
    Kokkos::View<data_type*, memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        slice_data( slice.data(), slice.numSoA() * slice.stride( 0 ) );

    // Extract the send buffer from the ghosted elements.
    std::size_t num_local = _comm_plan.numLocal();
    auto extract_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        std::size_t ghost_idx = i + num_local;
        auto s = SliceType::index_type::s( ghost_idx );
        auto a = SliceType::index_type::a( ghost_idx );
        std::size_t slice_offset = s * slice.stride( 0 ) + a;
        for ( std::size_t n = 0; n < num_comp; ++n )
            send_buffer( i, n ) =
                slice_data( slice_offset + SliceType::vector_length * n );
    };
    Kokkos::RangePolicy<ExecutionSpace> send_policy( 0, _send_size );
    Kokkos::parallel_for( "Cabana::scatter::extract_send_buffer", send_policy,
                          extract_send_buffer_func );
    Kokkos::fence();

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // for (std::size_t i = 0; i < send_buffer.extent(0); ++i)
    // {
    //     for ( std::size_t n = 0; n < num_comp; ++n )
    //         printf("R%d: send_buffer(%d, %d): %.1lf, dt size: %d\n", rank, i, n, (double)send_buffer(i, n), sizeof( data_type ));
    // }

    auto num_n = this->_comm_plan.numNeighbor();
    std::vector<int> send_counts( num_n ), recv_counts( num_n );
    std::vector<int> send_displs( num_n ), recv_displs( num_n );

    // printf("R%d: send/recv buffer bytes: %d, %d, num_comp: %d\n", rank,
    //     send_buffer.size()*sizeof( data_type ),
    //     recv_buffer.size()*sizeof( data_type ),
    //     num_comp);

    // Communicate data
    MPI_Status status;
    MPIL_Start(this->lrequest());
    MPIL_Wait(this->lrequest(), &status);
    
    // for (std::size_t i = 0; i < recv_buffer.extent(0); ++i)
    // {
    //     for ( std::size_t n = 0; n < num_comp; ++n )
    //         printf("R%d: recv_buffer(%d, %d): %.1lf\n", rank, i, n, (double)recv_buffer(i, n));
    // }

    // Get the steering vector for the sends.
    auto steering = _comm_plan.getExportSteering();

    // Scatter the ghosts in the receive buffer into the local values.
    auto scatter_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto s = SliceType::index_type::s( steering( i ) );
        auto a = SliceType::index_type::a( steering( i ) );
        std::size_t slice_offset = s * slice.stride( 0 ) + a;
        for ( std::size_t n = 0; n < num_comp; ++n )
            Kokkos::atomic_add(
                &slice_data( slice_offset + SliceType::vector_length * n ),
                recv_buffer( i, n ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> recv_policy( 0, _recv_size );
    Kokkos::parallel_for( "Cabana::scatter::apply::scatter_recv_buffer",
                          recv_policy, scatter_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( _comm_plan.comm() );
}

} // end namespace Cabana

#endif // end CABANA_HALO_LOCALITYAWARE_HPP
