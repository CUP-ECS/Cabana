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
  \file Cabana_CommunicationPlanMPIAdvance.hpp
  \brief Multi-node communication patterns.
  MPI Advance backend.
*/
#ifndef CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP
#define CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP

#include <Cabana_Utils.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>

#include <mpi.h>

#include <mpi_advance.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace Cabana
{
    
//---------------------------------------------------------------------------//
/*!
  \brief Communication plan class. Uses MPI Advance as the communication
  backend.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution will occur.

  The communication plan computes how to redistribute elements in a parallel
  data structure using MPI. Given a list of data elements on the local MPI
  rank and their destination ranks, the communication plan computes which rank
  each process is sending and receiving from and how many elements we will
  send and receive. In addition, it provides an export steering vector which
  describes how to pack the local data to be exported into contiguous send
  buffers for each destination rank (in the forward communication plan).

  Some nomenclature:

  Export - elements we are sending in the forward communication plan.

  Import - elements we are receiving in the forward communication plan.

  \note If a communication plan does self-sends (i.e. exports and imports data
  from its own ranks) then this data is first in the data structure. What this
  means is that neighbor 0 is the local rank and the data for that rank that
  is being exported will appear first in the steering vector.
*/
template <class MemorySpace>
class CommunicationPlan<MemorySpace, CommSpace::MPIAdvance>
    : public CommunicationPlanBase<MemorySpace>
{
  public:
    using typename CommunicationPlanBase<MemorySpace>::memory_space;
    using typename CommunicationPlanBase<MemorySpace>::execution_space;
    using typename CommunicationPlanBase<MemorySpace>::size_type;

  protected:
    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the CommunicationPlan is
      defined.
    */
    CommunicationPlan( MPI_Comm comm )
        : CommunicationPlanBase<MemorySpace>( comm )
    {
    }

    // The functions in the public block below would normally be protected but
    // we make them public to allow using private class data in CUDA kernels
    // with lambda functions.
  public:
    /*!
      \brief Neighbor and export rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param exec_space Kokkos execution space.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may be any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ExecutionSpace, class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExportsAndTopology( ExecutionSpace exec_space,
                                  const ViewType& element_export_ranks,
                                  const std::vector<int>& neighbor_ranks )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        // Store the number of export elements.
        this->_num_export_element = element_export_ranks.size();

        // Store the unique neighbors (this rank first).
        this->_neighbors = getUniqueTopology( this->comm(), neighbor_ranks );
        int num_n = this->_neighbors.size();

        // Create a neighbor communicator
        // Assume we send to and receive from each neighbor
        MPIX_Dist_graph_create_adjacent(this->comm(),
            num_n,
            this->_neighbors.data(), 
            MPI_UNWEIGHTED,
            num_n, 
            this->_neighbors.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0,
            &_neighbor_comm);

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( _neighbor_comm->global_comm, &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Initialize import/export sizes.
        this->_num_export.assign( num_n, 0 );
        this->_num_import.assign( num_n, 0 );

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Get the export counts.
        for ( int n = 0; n < num_n; ++n )
            this->_num_export[n] = neighbor_counts_host( this->_neighbors[n] );

        
        // Use MPIX_Neighbor_alltoallv_init to send number of exports to each neighbor.
        // This is an alltoall, not an alltoallv, but MPI Advance does not currently
        // have a Neighbor_alltoall.

        // Each send/recv is one int
        std::vector<int> sendcounts(num_n, 1);
        std::vector<int> recvcounts(num_n, 1);
        std::vector<int> sdispls(num_n);
        std::vector<int> rdispls(num_n);

        for (int i = 0; i < num_n; ++i)
        {
            sdispls[i] = i;
            rdispls[i] = i;
        }

        MPIX_Request* neighbor_request;
        MPIX_Info* xinfo;
        MPIX_Info_init(&xinfo);

        MPIX_Neighbor_alltoallv_init(
            this->_num_export.data(),  // sendbuffer
            sendcounts.data(),         // sendcounts
            sdispls.data(),            // sdispls
            MPI_UNSIGNED_LONG,         // sendtype
            this->_num_import.data(),  // recvbuffer
            recvcounts.data(),         // recvcounts
            rdispls.data(),            // rdispls
            MPI_UNSIGNED_LONG,         // recvtype
            _neighbor_comm,            // MPIX_Comm*
            xinfo,                     // MPIX_Info*
            &neighbor_request);        // output request

        MPI_Status status;
        MPIX_Start(neighbor_request);
        MPIX_Wait(neighbor_request, &status);
        MPIX_Request_free(&neighbor_request);

        // Get the total number of imports/exports.
        this->_total_num_export =
            std::accumulate( this->_num_export.begin(), this->_num_export.end(), 0 );
        this->_total_num_import =
            std::accumulate( this->_num_import.begin(), this->_num_import.end(), 0 );
        
        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( this->comm() );

        // Return the neighbor ids.
        return counts_and_ids.second;
    }

    /*!
      \brief Neighbor and export rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may be any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExportsAndTopology( const ViewType& element_export_ranks,
                                  const std::vector<int>& neighbor_ranks )
    {
        // Use the default execution space.
        return createFromExportsAndTopology(
            execution_space{}, element_export_ranks, neighbor_ranks );
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are sending to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param exec_space Kokkos execution space.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ExecutionSpace, class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExportsOnly( ExecutionSpace exec_space,
                           const ViewType& element_export_ranks )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        // Store the number of export elements.
        this->_num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        this->_neighbors.clear();
        this->_num_export.clear();
        this->_total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( neighbor_counts_host( r ) > 0 )
            {
                this->_neighbors.push_back( r );
                this->_num_export.push_back( neighbor_counts_host( r ) );
                this->_total_num_export += neighbor_counts_host( r );
                neighbor_counts_host( r ) = 1;
            }

        // Get the number of export ranks and initially allocate the import
        // sizes.
        int num_export_rank = this->_neighbors.size();
        this->_num_import.assign( num_export_rank, 0 );

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of imports to be the number of exports.
        bool self_send = false;
        for ( int n = 0; n < num_export_rank; ++n )
            if ( this->_neighbors[n] == my_rank )
            {
                std::swap( this->_neighbors[n], this->_neighbors[0] );
                std::swap( this->_num_export[n], this->_num_export[0] );
                // this->_num_import[0] = this->_num_export[0];
                self_send = true;
                break;
        
            }

        MPIX_Comm *xcomm;
        MPIX_Info *xinfo;
        MPIX_Comm_init(&xcomm, this->comm());
        MPIX_Info_init(&xinfo);

        int num_import_rank = -1;
        int *src;
        unsigned long* import_sizes;
        // std::vector<std::size_t> import_sizes( comm_size );
        MPIX_Alltoall_crs(num_export_rank, this->_neighbors.data(), 1, MPI_UNSIGNED_LONG, this->_num_export.data(),
                          &num_import_rank, &src, 1, MPI_UNSIGNED_LONG, (void**)&import_sizes,
                          xinfo, xcomm);
        
        MPIX_Info_free(&xinfo);
        MPIX_Comm_free(&xcomm);
        this->_total_num_import = std::accumulate(import_sizes, import_sizes + num_import_rank, 0UL);
            
        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Get the message source.
            const auto source = src[i];

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to.
            auto found_neighbor =
                std::find( this->_neighbors.begin(), this->_neighbors.end(), source );

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this.
            if ( found_neighbor == std::end( this->_neighbors ) )
            {
                this->_neighbors.push_back( source );
                this->_num_import.push_back( import_sizes[i] );
                this->_num_export.push_back( 0 );
            }

            // Otherwise if we already sent something to this neighbor that
            // means we already have a neighbor/export entry. Just assign the
            // import entry for that neighbor.
            else
            {
                auto n = std::distance( this->_neighbors.begin(), found_neighbor );
                this->_num_import[n] = import_sizes[i];
            }
        }

        MPIX_Free(src);
        MPIX_Free(import_sizes);

        // Now that we know our neighbors, create a neighbor communicator
        // to optimize calls to Cabana::migrate.
        MPIX_Dist_graph_create_adjacent(this->comm(),
            this->_neighbors.size(),
            this->_neighbors.data(), 
            MPI_UNWEIGHTED,
            this->_neighbors.size(), 
            this->_neighbors.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0,
            &_neighbor_comm);

        // Return the neighbor ids.
        return counts_and_ids.second;
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are sending to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExportsOnly( const ViewType& element_export_ranks )
    {
        // Use the default execution space.
        return createFromExportsOnly( execution_space{}, element_export_ranks );
    }

    /*!
      \brief Neighbor and import rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param exec_space Kokkos execution space.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ExecutionSpace, class ViewType>
    auto createFromImportsAndTopology( ExecutionSpace exec_space,
                                       const ViewType& element_import_ranks,
                                       const ViewType& element_import_ids,
                                       const std::vector<int>& neighbor_ranks )
        -> std::tuple<Kokkos::View<typename ViewType::size_type*,
                                   typename ViewType::memory_space>,
                      Kokkos::View<int*, typename ViewType::memory_space>,
                      Kokkos::View<int*, typename ViewType::memory_space>>
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        if ( element_import_ids.size() != element_import_ranks.size() )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Store the unique neighbors (this rank first).
        this->_neighbors = getUniqueTopology( this->comm(), neighbor_ranks );
        int num_n = this->_neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Initialize import/export sizes.
        this->_num_export.assign( num_n, 0 );
        this->_num_import.assign( num_n, 0 );

        // Count the number of imports this rank needs from other ranks. Keep
        // track of which slot we get in our neighbor's send buffer?
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_import_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Get the import counts.
        for ( int n = 0; n < num_n; ++n )
            this->_num_import[n] = neighbor_counts_host( this->_neighbors[n] );

        // Post receives to get the number of indices I will send to each rank.
        // Post that many wildcard recieves to get the number of indices I will
        // send to each rank
        std::vector<MPI_Request> requests;
        requests.reserve( num_n * 2 );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &this->_num_export[n], 1, MPI_UNSIGNED_LONG, this->_neighbors[n],
                           mpi_tag, this->comm(), &( requests.back() ) );
            }
            else // Self import
            {
                this->_num_export[n] = this->_num_import[n];
            }

        // Send the number of imports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Isend( &this->_num_import[n], 1, MPI_UNSIGNED_LONG, this->_neighbors[n],
                           mpi_tag, this->comm(), &( requests.back() ) );
            }

        // Wait on messages.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Get the total number of imports/exports.
        this->_total_num_export =
            std::accumulate( this->_num_export.begin(), this->_num_export.end(), 0 );
        this->_total_num_import =
            std::accumulate( this->_num_import.begin(), this->_num_import.end(), 0 );
        this->_num_export_element = this->_total_num_export;

        // Post receives to get the indices other processes are requesting
        // i.e. our export indices
        Kokkos::View<int*, memory_space> export_indices( "export_indices",
                                                         this->_total_num_export );
        size_t idx = 0;
        int num_messages = this->_total_num_export + element_import_ranks.extent( 0 );
        std::vector<MPI_Request> mpi_requests( num_messages );
        std::vector<MPI_Status> mpi_statuses( num_messages );
        for ( int i = 0; i < num_n; i++ )
        {
            for ( int j = 0; j < this->_num_export[i]; j++ )
            {
                MPI_Irecv( export_indices.data() + idx, 1, MPI_INT,
                           this->_neighbors[i], mpi_tag, this->comm(), &mpi_requests[idx] );
                idx++;
            }
        }

        // Send the indices we need
        for ( size_t i = 0; i < element_import_ranks.extent( 0 ); i++ )
        {
            MPI_Isend( element_import_ids.data() + i, 1, MPI_INT,
                       *( element_import_ranks.data() + i ), mpi_tag, this->comm(),
                       &mpi_requests[idx++] );
        }

        // Wait for all count exchanges to complete
        const int ec1 = MPI_Waitall( num_messages, mpi_requests.data(),
                                     mpi_statuses.data() );
        if ( MPI_SUCCESS != ec1 )
            throw std::logic_error( "Failed MPI Communication" );

        // Now, build the export steering
        // Export rank in mpi_statuses[i].MPI_SOURCE
        // Export ID in export_indices(i)
        Kokkos::View<int*, Kokkos::HostSpace> element_export_ranks_h(
            "element_export_ranks_h", this->_total_num_export );
        for ( size_t i = 0; i < this->_total_num_export; i++ )
        {
            element_export_ranks_h[i] = mpi_statuses[i].MPI_SOURCE;
        }
        auto element_export_ranks = Kokkos::create_mirror_view_and_copy(
            memory_space(), element_export_ranks_h );

        auto counts_and_ids2 = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy indices_send to device mempry before returning
        // auto export_indices_d = Kokkos::create_mirror_view_and_copy(
        //     memory_space(), export_indices );

        // Return the neighbor ids, export ranks, and export indices
        return std::tuple{ counts_and_ids2.second, element_export_ranks,
                           export_indices };
    }

    /*!
      \brief Neighbor and import rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ViewType>
    auto createFromImportsAndTopology( const ViewType& element_import_ranks,
                                       const ViewType& element_import_ids,
                                       const std::vector<int>& neighbor_ranks )
    {
        // Use the default execution space.
        return createFromImportsAndTopology(
            execution_space{}, element_import_ranks, element_import_ids,
            neighbor_ranks );
    }

    /*!
      \brief Import rank creator. Use this when you don't know who you will
      be receiving from - only who you are importing from. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \param exec_space Kokkos execution space.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ExecutionSpace, class ViewType>
    auto createFromImportsOnly( ExecutionSpace exec_space,
                                const ViewType& element_import_ranks,
                                const ViewType& element_import_ids )
        -> std::tuple<Kokkos::View<typename ViewType::size_type*,
                                   typename ViewType::memory_space>,
                      Kokkos::View<int*, typename ViewType::memory_space>,
                      Kokkos::View<int*, typename ViewType::memory_space>>
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        if ( element_import_ids.size() != element_import_ranks.size() )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int rank = -1;
        MPI_Comm_rank( this->comm(), &rank );

        this->_total_num_import = element_import_ranks.extent(0);

        // Step 1: Initialize indices
        Kokkos::View<int*, memory_space> indices("indices", this->_total_num_import);
        Kokkos::parallel_for("InitIndices", Kokkos::RangePolicy<ExecutionSpace>( 0, this->_total_num_import ), KOKKOS_LAMBDA(int i) {
            indices(i) = i;
        });
    
        // Step 2: Set up bin sort
        using BinOp = Kokkos::BinOp1D<Kokkos::View<int*, memory_space>>;
        BinOp bin_op(comm_size, 0, comm_size-1);
        Kokkos::BinSort<Kokkos::View<int*, memory_space>, BinOp> bin_sort(element_import_ranks, bin_op, true);
    
        // Step 3: Sort indices
        bin_sort.create_permute_vector();
        bin_sort.sort(indices);
    
        // Step 4: Permute both arrays
        Kokkos::View<int*, memory_space> ranks_sorted("ranks_sorted", this->_total_num_import);
        Kokkos::View<int*, memory_space> ids_sorted("ids_sorted", this->_total_num_import);
        Kokkos::parallel_for("PermuteExports", Kokkos::RangePolicy<ExecutionSpace>( 0, this->_total_num_import ), KOKKOS_LAMBDA(int i) {
            int sorted_i = indices(i);
            ranks_sorted(i) = element_import_ranks(sorted_i);
            ids_sorted(i)   = element_import_ids(sorted_i);
        });

        // Count the number of imports this rank needs from other ranks. Keep
        // track of which slot we get in our neighbor's send buffer?
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, ranks_sorted, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Clear vectors before we use them
        this->_neighbors.clear();
        this->_num_export.clear();
        this->_num_import.clear();

        for ( size_t i = 0; i < neighbor_counts_host.extent( 0 ); i++ )
        {
            if ( neighbor_counts_host( i ) != 0 )
            {
                // Store we are importing this many from this rank
                this->_neighbors.push_back( i );
                this->_num_import.push_back( neighbor_counts_host( i ) );
            }
        }

        // Store offsets into ranks_sorted to send
        std::vector<int> sdispls;
        sdispls.push_back(0);
        for (int neighbor_rank : this->_neighbors)
            sdispls.push_back(sdispls.back() + neighbor_counts_host(neighbor_rank));
        
        // Store send counts to each rank
        std::vector<int> sendcounts;
        for (int neighbor_rank : this->_neighbors)
            sendcounts.push_back(neighbor_counts_host(neighbor_rank));


        // Assign all exports to zero
        this->_num_export.assign( this->_num_import.size(), 0 );


        MPIX_Comm *xcomm;
        MPIX_Info *xinfo;
        MPIX_Comm_init(&xcomm, this->comm());
        MPIX_Info_init(&xinfo);

        int num_export_rank = -1, total_num_export = -1;
        int *src, *recv_counts, *recv_displs, *recv_vals;
        // std::vector<std::size_t> import_sizes( comm_size );
        auto ids_sorted_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ids_sorted);
        MPIX_Alltoallv_crs(this->_neighbors.size(), this->_total_num_import, this->_neighbors.data(), sendcounts.data(), 
            sdispls.data(), MPI_INT, ids_sorted_host.data(),
            &num_export_rank, &total_num_export, &src, &recv_counts,
            &recv_displs, MPI_INT, (void**)&recv_vals, xinfo, xcomm); 
        
        
        this->_total_num_export = total_num_export;
        
        MPIX_Info_free(&xinfo);
        MPIX_Comm_free(&xcomm);
            
        // Save ranks we got messages from and track total messages to size
        // buffers
        // Track which ranks we received data from
        for (int i = 0; i < num_export_rank; ++i)
        {
            int exporting_rank = src[i];
            int count = recv_counts[i];

            // Check if this is a new neighbor or an existing one (due to import)
            auto found = std::find(this->_neighbors.begin(), this->_neighbors.end(), exporting_rank);
            if (found == this->_neighbors.end())
            {
                this->_neighbors.push_back(exporting_rank);
                this->_num_import.push_back(0);
                this->_num_export.push_back(count);
            }
            else
            {
                int n = std::distance(this->_neighbors.begin(), found);
                this->_num_export[n] = count;
            }
        }

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of exports to be the number of imports.
        bool self_send = false;
        for ( int n = 0; n < this->_neighbors.size(); ++n )
            if ( this->_neighbors[n] == rank )
            {
                std::swap( this->_neighbors[n], this->_neighbors[0] );
                std::swap( this->_num_export[n], this->_num_export[0] );
                std::swap( this->_num_import[n], this->_num_import[0] );
                this->_num_export[0] = this->_num_import[0];
                self_send = true;
                break;
            }

        // Total number of imports and exports are now known
        this->_num_export_element = this->_total_num_export;

        // Now, build the export steering
        Kokkos::View<int*, Kokkos::HostSpace> element_export_ranks_h(
            "element_export_ranks", this->_total_num_export );
        Kokkos::View<int*, Kokkos::HostSpace> export_indices_h(
            "export_indices", this->_total_num_export );

        // Fill the export ranks and indices
        int offset = 0;
        for (int i = 0; i < num_export_rank; ++i)
        {
            int dest_rank = src[i];           // The rank to send to
            int count = recv_counts[i];       // How many elements to send
            int disp = recv_displs[i];        // Where they are in recv_vals

            for (int j = 0; j < count; ++j)
            {
                element_export_ranks_h(offset) = dest_rank;
                export_indices_h(offset) = recv_vals[disp + j];
                ++offset;
            }
        }

        MPIX_Free(src);
        MPIX_Free(recv_counts);
        MPIX_Free(recv_displs);
        MPIX_Free(recv_vals);

        auto element_export_ranks = Kokkos::create_mirror_view_and_copy(
            memory_space(), element_export_ranks_h );
        auto export_indices = Kokkos::create_mirror_view_and_copy(
            memory_space(), export_indices_h );

        auto counts_and_ids2 = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        return std::tuple{ counts_and_ids2.second, element_export_ranks,
                           export_indices };
    }

    /*!
      \brief Import rank creator. Use this when you don't know who you will
      be receiving from - only who you are importing from. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ViewType>
    auto createFromImportsOnly( const ViewType& element_import_ranks,
                                const ViewType& element_import_ids )
    {
        // Use the default execution space.
        return createFromImportsOnly( execution_space{}, element_import_ranks,
                                      element_import_ids );
    }

  private:
    // Variables needed only for MPI Advance
    MPIX_Comm* _neighbor_comm;
};

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP
