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
  \file Cabana_Halo_MpiAdvance.hpp
  \brief Multi-node particle scatter/gather, MpiAdvance implementations
*/
#ifndef CABANA_HALO_MPIADVANCE_HPP
#define CABANA_HALO_MPIADVANCE_HPP

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
std::enable_if_t<std::is_same<CommSpaceType, CommSpace::MpiAdvance>::value,
                 void>
Gather<HaloType, AoSoAType,
       typename std::enable_if<is_aosoa<AoSoAType>::value>::type>::
    applyImpl( ExecutionSpace, CommSpaceType )
{
       Kokkos::Profiling::ScopedRegion region( "Cabana::gather" );

        // Get the buffers and particle data (local copies for lambdas below).
        auto send_buffer = this->getSendBuffer();

        auto aosoa = this->getData();
        int num_n = _halo.numNeighbor();
        int my_rank = -1;
        MPI_Comm_rank( _halo.comm(), &my_rank );

        std::size_t num_stay =
                ( num_n > 0 && _halo.neighborRank( 0 ) == my_rank )
                    ? _halo.numExport( 0 )
                    : 0;





        // Get the steering vector for the sends.
        auto steering = _halo.getExportSteering();
        // Gather from the local data into a tuple-contiguous send buffer.
        auto gather_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            send_buffer( i ) = aosoa.getTuple( steering( i ) );
        };
        Kokkos::RangePolicy<ExecutionSpace> send_policy( 0, _send_size );
        Kokkos::parallel_for( "Cabana::gather::gather_send_buffer", send_policy,
                              gather_send_buffer_func );
        Kokkos::fence();


    this->buff_size =sizeof( buffer_type );
    if (!this->setup_persistent) {
        this->setupPersistent(_halo);
    }

    MPI_Status status;
    MPIX_Start(*(this->neighbor_request));
    MPIX_Wait(*(this->neighbor_request), &status);

    auto recv_buffer = this->getReceiveBuffer();

        // Extract the receive buffer into the ghosted elements.
        std::size_t num_local = _halo.numLocal();
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
        MPI_Barrier( _halo.comm() );
}

/*!
    \brief Perform the gather operation.
*/
template <class HaloType, class SliceType>
template <class ExecutionSpace, class CommSpaceType>
std::enable_if_t<std::is_same<CommSpaceType, CommSpace::MpiAdvance>::value,
                 void>
Gather<HaloType, SliceType,
       typename std::enable_if<is_slice<SliceType>::value>::type>::
    applyImpl( ExecutionSpace, CommSpaceType )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::gather" );

    // Get the buffers (local copies for lambdas below).
    auto send_buffer = this->getSendBuffer();
    auto recv_buffer = this->getReceiveBuffer();
    auto slice = this->getData();

    // Get the number of components in the slice.
    std::size_t num_comp = this->getSliceComponents();

    // Get the raw slice data.
    auto slice_data = slice.data();

    // Get the steering vector for the sends.
    auto steering = _halo.getExportSteering();

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
    this->buff_size =num_comp * sizeof( typename SliceType::value_type );
    if (!this->setup_persistent) {
        this->setupPersistent(_halo);
    }

    MPI_Status status;
    MPIX_Start(*(this->neighbor_request));
    MPIX_Wait(*(this->neighbor_request), &status);




    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = _halo.numLocal();
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
    MPI_Barrier( _halo.comm() );
}

/**********
 * SCATTER *
 **********/

template <class HaloType, class SliceType>
template <class ExecutionSpace, class CommSpaceType>
std::enable_if_t<std::is_same<CommSpaceType, CommSpace::MpiAdvance>::value,
                 void>
Scatter<HaloType, SliceType>::applyImpl( ExecutionSpace, CommSpaceType )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::scatter" );

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
    std::size_t num_local = _halo.numLocal();
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

    this->buff_size =num_comp * sizeof( typename SliceType::value_type );
    if (!this->setup_persistent) {
        this->setupPersistent(_halo);
    }

    MPI_Status status;
    MPIX_Start(*(this->neighbor_request));
    MPIX_Wait(*(this->neighbor_request), &status);

    // Get the steering vector for the sends.
    auto steering = _halo.getExportSteering();

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
    MPI_Barrier( _halo.comm() );
}

} // end namespace Cabana

#endif // end CABANA_HALO_MPIADVANCE_HPP
