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
  \file Cabana_Migrate_MPI.hpp
  \brief MPI Advance implementation of Cabana::migrate variations
*/
#ifndef CABANA_MIGRATE_MPIADVANCE_HPP
#define CABANA_MIGRATE_MPIADVANCE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <mpi_advance.h>

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
template <class ExecutionSpace, class Migrator_t, class AoSoA_t>
void migrateData( CommSpace::MPIAdvance,
    ExecutionSpace, const Migrator_t& migrator, const AoSoA_t& src,
    AoSoA_t& dst,
    typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                is_collector<Migrator_t>::value ) &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::migrate" );

    static_assert(
        is_accessible_from<typename Migrator_t::memory_space, ExecutionSpace>{},
        "" );

    int offset = 0;
    if constexpr ( is_collector<Migrator_t>::value )
    {
        // If src and dest are the same, the Collector places
        // collected data at the end of the AoSoA instead of
        // overwriting the existing data like a Distributor.
        if ( src.data() == dst.data() )
            offset = migrator.numOwned();
    }

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( migrator.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = migrator.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && migrator.neighborRank( 0 ) == my_rank )
            ? migrator.numExport( 0 )
            : 0;

    // Allocate a send buffer.
    std::size_t num_send = migrator.totalNumExport() - num_stay;
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Migrator_t::memory_space>
        send_buffer("migrator_send_buffer", num_send);
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Migrator_t::memory_space>
        recv_buffer("migrator_recv_buffer", migrator.totalNumImport());

    // Get the steering vector for the sends.
    auto steering = migrator.getExportSteering();
    Kokkos::parallel_for(
        "Cabana::Impl::migrateData::build_send_buffer",
        Kokkos::RangePolicy<ExecutionSpace>(0, migrator.totalNumExport()),
        KOKKOS_LAMBDA(const std::size_t i) {
            auto tpl = src.getTuple(steering(i));
            if (i < num_stay)
                recv_buffer(i) = tpl;
            else
                send_buffer(i - num_stay) = tpl;
        });
    Kokkos::fence();

    // Build counts and displacements
    std::vector<int> send_counts(num_n), recv_counts(num_n);
    std::vector<int> send_displs(num_n), recv_displs(num_n);

    std::size_t send_offset = 0, recv_offset = 0;
    for (int n = 0; n < num_n; ++n)
    {
        recv_counts[n] = migrator.numImport(n) * sizeof(typename AoSoA_t::tuple_type);
        recv_displs[n] = recv_offset;
        recv_offset += recv_counts[n];

        if (migrator.neighborRank(n) == my_rank)
        {
            send_counts[n] = 0;
            send_displs[n] = 0;
        }
        else
        {
            send_counts[n] = migrator.numExport(n) * sizeof(typename AoSoA_t::tuple_type);
            send_displs[n] = send_offset;
            send_offset += send_counts[n];
        }
    }

    // MPI Advance does not currently support GPU communication,
    // so buffers need to be copied to host memory
    auto send_buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_buffer);
    auto recv_buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv_buffer);

    MPI_Datatype datatype = MPI_BYTE;
    auto xcomm = migrator.xcomm();

    MPIX_Request* neighbor_request;
    MPIX_Info* xinfo;
    MPIX_Info_init( &xinfo );

    MPIX_Neighbor_alltoallv_init(send_buffer.data(), send_counts.data(), send_displs.data(), datatype,
                                 recv_buffer.data(), recv_counts.data(), recv_displs.data(), datatype,
                                 xcomm, xinfo, &neighbor_request);

    MPI_Status status;
    MPIX_Start( neighbor_request );
    MPIX_Wait( neighbor_request, &status );
    MPIX_Request_free( &neighbor_request );
    MPIX_Info_free( &xinfo );

    // Copy recv buffer back to device memory
    recv_buffer = Kokkos::create_mirror_view_and_copy(typename Migrator_t::memory_space(), recv_buffer_h);

    Kokkos::parallel_for(
        "Cabana::Impl::migrateData::extract_recv_buffer",
        Kokkos::RangePolicy<ExecutionSpace>(0, migrator.totalNumImport()),
        KOKKOS_LAMBDA(const std::size_t i) {
            dst.setTuple(offset + i, recv_buffer(i));
        });
    Kokkos::fence();

    MPI_Barrier(migrator.comm());
}


//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the migrator forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Migrator_t - Migrator type - must be a Distributor or a Collector.
  \tparam Slice_t Slice type - must be a Slice.

  \param migrator The migrator to use for the migration.
  \param src The slice containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the migrator.
  \param dst The slice to which the migrated data will be written. Must be the
  same size as the number of imports given by the migrator on this
  rank. Call totalNumImport() on the migrator to get this size value.
*/
template <class ExecutionSpace, class Migrator_t, class Slice_t>
void migrateSlice( CommSpace::MPIAdvance,
    ExecutionSpace, const Migrator_t& migrator, const Slice_t& src,
    Slice_t& dst,
    typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                is_collector<Migrator_t>::value ) &&
                              is_slice<Slice_t>::value ),
                            int>::type* = 0 )
{
    // Check that dst is the right size.
    if ( dst.size() != migrator.totalNumImport() )
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
    MPI_Comm_rank( migrator.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = migrator.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && migrator.neighborRank( 0 ) == my_rank )
            ? migrator.numExport( 0 )
            : 0;

    // Allocate a send buffer. Note this one is layout right so the components
    // of each element are consecutive in memory.
    std::size_t num_send = migrator.totalNumExport() - num_stay;
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Migrator_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_send_buffer" ),
            num_send, num_comp );

    // Allocate a receive buffer. Note this one is layout right so the
    // components of each element are consecutive in memory.
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Migrator_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_recv_buffer" ),
            migrator.totalNumImport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = migrator.getExportSteering();
    Kokkos::parallel_for(
        "Cabana::migrate::build_send_buffer",
        Kokkos::RangePolicy<ExecutionSpace>( 0, migrator.totalNumExport() ),
        KOKKOS_LAMBDA( const std::size_t i )
        {
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

    std::vector<int> send_counts(num_n), recv_counts(num_n);
    std::vector<int> send_displs(num_n), recv_displs(num_n);

    std::size_t send_offset = 0, recv_offset = 0;
    for ( int n = 0; n < num_n; ++n )
    {
        recv_counts[n] = migrator.numImport(n) * num_comp *
                         sizeof(typename Slice_t::value_type);
        recv_displs[n] = recv_offset;
        recv_offset += recv_counts[n];

        if ( migrator.neighborRank(n) == my_rank )
        {
            send_counts[n] = 0;
            send_displs[n] = 0;
        }
        else
        {
            send_counts[n] = migrator.numExport(n) * num_comp *
                             sizeof(typename Slice_t::value_type);
            send_displs[n] = send_offset;
            send_offset += send_counts[n];
        }
    }

    // MPI Advance does not currently support GPU communication,
    // so buffers need to be copied to host memory
    auto send_buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_buffer);
    auto recv_buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv_buffer);

    MPI_Datatype datatype = MPI_BYTE;
    auto xcomm = migrator.xcomm();

    MPIX_Request* neighbor_request;
    MPIX_Info* xinfo;
    MPIX_Info_init( &xinfo );

    MPIX_Neighbor_alltoallv_init(send_buffer_h.data(), send_counts.data(), send_displs.data(), datatype,
                                 recv_buffer_h.data(), recv_counts.data(), recv_displs.data(), datatype,
                                 xcomm, xinfo, &neighbor_request);

    MPI_Status status;
    MPIX_Start( neighbor_request );
    MPIX_Wait( neighbor_request, &status );
    MPIX_Request_free( &neighbor_request );
    MPIX_Info_free( &xinfo );

    // Copy recv buffer back to device memory
    recv_buffer = Kokkos::create_mirror_view_and_copy(typename Migrator_t::memory_space(), recv_buffer_h);

    Kokkos::parallel_for(
        "Cabana::migrate::extract_recv_buffer",
        Kokkos::RangePolicy<ExecutionSpace>( 0, migrator.totalNumImport() ),
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( i );
            auto a = Slice_t::index_type::a( i );
            std::size_t dst_offset = s * dst.stride( 0 ) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                dst_data[dst_offset + n * Slice_t::vector_length] =
                    recv_buffer( i, n );
        } );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( migrator.comm() );
}


//---------------------------------------------------------------------------//
//! \endcond
} // end namespace Impl

} // end namespace Cabana

#endif // CABANA_MIGRATE_MPIADVANCE_HPP
