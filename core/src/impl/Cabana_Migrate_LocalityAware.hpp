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
  \file Cabana_Migrate_LocalityAware.hpp
  \brief MPI Advance implementation of Cabana::migrate variations
*/
#ifndef CABANA_MIGRATE_LOCALITYAWARE_HPP
#define CABANA_MIGRATE_LOCALITYAWARE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <locality_aware.h>

#include <exception>
#include <vector>

//---------------------------------------------------------------------------//
namespace Cabana
{

namespace Impl
{
//! \cond Impl
//---------------------------------------------------------------------------//
// Synchronously move data between a source and destination AoSoA by executing
// the forward communication plan.
template <class ExecutionSpace, class Distributor_t, class AoSoA_t>
void migrateData(
    LocalityAware, ExecutionSpace, const Distributor_t& distributor,
    const AoSoA_t& src, AoSoA_t& dst,
    typename std::enable_if<( ( is_distributor<Distributor_t>::value ) &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::migrate (LocalityAware)" );

    static_assert( is_accessible_from<typename Distributor_t::memory_space,
                                      ExecutionSpace>{},
                   "" );

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( distributor.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = distributor.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && distributor.neighborRank( 0 ) == my_rank )
            ? distributor.numExport( 0 )
            : 0;

    // Allocate a send buffer.
    std::size_t num_send = distributor.totalNumExport() - num_stay;
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Distributor_t::memory_space>
        send_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_send_buffer" ),
                     distributor.totalNumExport() );

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Distributor_t::memory_space>
        recv_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_recv_buffer" ),
                     distributor.totalNumImport() );

    // Get the steering vector for the sends.
    auto steering = distributor.getExportSteering();

    // Gather the exports from the source AoSoA into the tuple-contiguous send
    // buffer or the receive buffer if the data is staying. We know that the
    // steering vector is ordered such that the data staying on this rank
    // comes first.
    auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto tpl = src.getTuple( steering( i ) );
        if ( i < num_stay )
            recv_buffer( i ) = tpl;
        else
            send_buffer( i - num_stay ) = tpl;
    };
    Kokkos::RangePolicy<ExecutionSpace> build_send_buffer_policy(
        0, distributor.totalNumExport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::build_send_buffer",
                          build_send_buffer_policy, build_send_buffer_func );
    Kokkos::fence();

    // MPI Advance does not currently support GPU communication,
    // so buffers must be copied to host memory
    auto send_buffer_h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), send_buffer );
    auto recv_buffer_h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), recv_buffer );

    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_displs;
    std::vector<int> send_neighbors;
    std::vector<int> recv_neighbors;

    using value_type = typename AoSoA_t::tuple_type;
    const std::size_t value_bytes = sizeof(value_type);

    // --- Receive side ---
    std::size_t recv_offset = 0;
    for (int n = 0; n < num_n; ++n)
    {
        const int num_import = distributor.numImport(n);
        const int neighbor   = distributor.neighborRank(n);

        if (num_import == 0)
            continue; // skip zero-length messages

        recv_neighbors.push_back(neighbor);
        recv_counts.push_back(static_cast<int>(num_import * value_bytes));
        recv_displs.push_back(static_cast<int>(recv_offset * value_bytes));

        recv_offset += num_import;
    }

    // --- Send side ---
    std::size_t send_offset = 0;
    for (int n = 0; n < num_n; ++n)
    {
        const int num_export = distributor.numExport(n);
        const int neighbor   = distributor.neighborRank(n);

        if (num_export == 0)
            continue; // skip zero-length messages

        send_neighbors.push_back(neighbor);
        send_counts.push_back(static_cast<int>(num_export * value_bytes));
        send_displs.push_back(static_cast<int>(send_offset * value_bytes));

        send_offset += num_export;
    }


    for (int i = 0; i < send_counts.size(); i++)
    {
        printf("R%d: sc,d: (%d, %d), R%d (cp: %d), num_send: %d\n", my_rank,
            send_counts[i] / value_bytes, send_displs[i],
            send_neighbors[i], distributor.nonZeroSendNeighbor(i),
            send_buffer.extent(0));
    }
    for (int i = 0; i < recv_counts.size(); i++)
    {
        printf("R%d: rc,d: (%d, %d), R%d (cp: %d), num_recv: %d\n", my_rank,
            recv_counts[i] / value_bytes, recv_displs[i],
            recv_neighbors[i], distributor.nonZeroRecvNeighbor(i),
            recv_buffer.extent(0) );
    }

    auto lcomm = distributor.lcomm();
    auto ltopo = distributor.ltopo();
    auto linfo = distributor.linfo();

    MPIL_Request* neighbor_request;
    MPIL_Neighbor_alltoallv_init_topo(
        send_buffer_h.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
        recv_buffer_h.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
        ltopo, lcomm, linfo, &neighbor_request );

    MPI_Status status;
    MPIL_Start( neighbor_request );
    MPIL_Wait( neighbor_request, &status );
    MPIL_Request_free( &neighbor_request );

    // Copy data back to device
    Kokkos::deep_copy(recv_buffer, recv_buffer_h);
    // recv_buffer = Kokkos::create_mirror_view_and_copy(
    //     typename Distributor_t::memory_space(), recv_buffer_h );

    // Extract the receive buffer into the destination AoSoA.
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        dst.setTuple( i, recv_buffer( i ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> extract_recv_buffer_policy(
        0, distributor.totalNumImport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    auto slice_int_dst_host = Cabana::slice<0>( dst );
    for (int i = 0; i < dst.size(); ++i)
    {
        printf("R%d: dst(%d): %d\n", my_rank, i, slice_int_dst_host(i));
    }

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( distributor.comm() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Distributor_t - Distributor type - must be a Distributor
  \tparam Slice_t Slice type - must be a Slice.

  \param distributor The distributor to use for the migration.
  \param src The slice containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.
  \param dst The slice to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class ExecutionSpace, class Distributor_t, class Slice_t>
void migrateSlice(
    LocalityAware, ExecutionSpace, const Distributor_t& distributor,
    const Slice_t& src, Slice_t& dst,
    typename std::enable_if<( ( is_distributor<Distributor_t>::value ) &&
                              is_slice<Slice_t>::value ),
                            int>::type* = 0 )
{
    // Check that dst is the right size.
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error(
            "migrateSlice: Destination is the wrong size for migration!" );

    // Get the number of components in the slices.
    size_t num_comp = 1;
    for ( size_t d = 2; d < src.viewRank(); ++d )
        num_comp *= src.extent( d );

    // Get the raw slice data.
    auto src_data = src.data();
    auto dst_data = dst.data();

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( distributor.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = distributor.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && distributor.neighborRank( 0 ) == my_rank )
            ? distributor.numExport( 0 )
            : 0;

    // Allocate a send buffer. Note this one is layout right so the components
    // of each element are consecutive in memory.
    std::size_t num_send = distributor.totalNumExport() - num_stay;
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Distributor_t::memory_space>
        send_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_send_buffer" ),
                     num_send, num_comp );

    // Allocate a receive buffer. Note this one is layout right so the
    // components of each element are consecutive in memory.
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Distributor_t::memory_space>
        recv_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_recv_buffer" ),
                     distributor.totalNumImport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = distributor.getExportSteering();
    Kokkos::parallel_for(
        "Cabana::migrate::build_send_buffer",
        Kokkos::RangePolicy<ExecutionSpace>( 0, distributor.totalNumExport() ),
        KOKKOS_LAMBDA( const std::size_t i ) {
            auto s_src = Slice_t::index_type::s( steering( i ) );
            auto a_src = Slice_t::index_type::a( steering( i ) );
            std::size_t src_offset = s_src * src.stride( 0 ) + a_src;
            if ( i < num_stay )
            {
                for ( std::size_t n = 0; n < num_comp; ++n )
                    recv_buffer( i, n ) =
                        src_data[src_offset + n * Slice_t::vector_length];
            }
            else
            {
                for ( std::size_t n = 0; n < num_comp; ++n )
                    send_buffer( i - num_stay, n ) =
                        src_data[src_offset + n * Slice_t::vector_length];
            }
        } );
    Kokkos::fence();

    std::vector<int> send_counts( num_n ), recv_counts( num_n );
    std::vector<int> send_displs( num_n ), recv_displs( num_n );

    std::size_t send_offset = 0, recv_offset = 0;
    for ( int n = 0; n < num_n; ++n )
    {
        recv_counts[n] = distributor.numImport( n ) * num_comp *
                         sizeof( typename Slice_t::value_type );
        recv_displs[n] = recv_offset;
        recv_offset += recv_counts[n];

        if ( distributor.neighborRank( n ) == my_rank )
        {
            send_counts[n] = 0;
            send_displs[n] = 0;
        }
        else
        {
            send_counts[n] = distributor.numExport( n ) * num_comp *
                             sizeof( typename Slice_t::value_type );
            send_displs[n] = send_offset;
            send_offset += send_counts[n];
        }
    }

    // MPI Advance does not currently support GPU communication,
    // so buffers need to be copied to host memory
    auto send_buffer_h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), send_buffer );
    auto recv_buffer_h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), recv_buffer );

    MPI_Datatype datatype = MPI_BYTE;
    auto xcomm = distributor.lcomm();
    auto xtopo = distributor.ltopo();

    MPIL_Request* neighbor_request;
    MPIL_Info* xinfo;
    MPIL_Info_init( &xinfo );

    MPIL_Neighbor_alltoallv_init_topo(
        send_buffer_h.data(), send_counts.data(), send_displs.data(), datatype,
        recv_buffer_h.data(), recv_counts.data(), recv_displs.data(), datatype,
        xtopo, xcomm, xinfo, &neighbor_request );

    MPI_Status status;
    MPIL_Start( neighbor_request );
    MPIL_Wait( neighbor_request, &status );
    MPIL_Request_free( &neighbor_request );
    MPIL_Info_free( &xinfo );

    // Copy recv buffer back to device memory
    recv_buffer = Kokkos::create_mirror_view_and_copy(
        typename Distributor_t::memory_space(), recv_buffer_h );

    Kokkos::parallel_for(
        "Cabana::migrate::extract_recv_buffer",
        Kokkos::RangePolicy<ExecutionSpace>( 0, distributor.totalNumImport() ),
        KOKKOS_LAMBDA( const std::size_t i ) {
            auto s = Slice_t::index_type::s( i );
            auto a = Slice_t::index_type::a( i );
            std::size_t dst_offset = s * dst.stride( 0 ) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                dst_data[dst_offset + n * Slice_t::vector_length] =
                    recv_buffer( i, n );
        } );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( distributor.comm() );
}

//---------------------------------------------------------------------------//
//! \endcond
} // end namespace Impl

} // end namespace Cabana

#endif // CABANA_MIGRATE_LOCALITYAWARE_HPP
