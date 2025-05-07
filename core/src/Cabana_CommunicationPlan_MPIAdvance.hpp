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
  \file Cabana_CommunicationPlan.hpp
  \brief Multi-node communication patterns
*/
#ifndef CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP
#define CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP

#include <Cabana_Utils.hpp>

#include <Cabana_CommunicationPlan.hpp>

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
  \brief Communication plan base class.

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
class CommunicationPlan<MemorySpace, CommSpaces::MPIAdvance>
{
  public:
    // FIXME: extracting the self type for backwards compatibility with previous
    // template on DeviceType. Should simply be MemorySpace after next release.
    //! Memory space.
    using memory_space = typename MemorySpace::memory_space;
    // FIXME: replace warning with memory space assert after next release.
    static_assert( Impl::deprecated( Kokkos::is_device<MemorySpace>() ) );

    //! Default device type.
    using device_type [[deprecated]] = typename memory_space::device_type;

    //! Default execution space.
    using execution_space = typename memory_space::execution_space;

    // FIXME: extracting the self type for backwards compatibility with previous
    // template on DeviceType. Should simply be memory_space::size_type after
    // next release.
    //! Size type.
    using size_type = typename memory_space::memory_space::size_type;

    //! Communication plan type
    using plan_type = CommSpaces::MPIAdvance;

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the distributor is defined.
    */
    CommunicationPlan( MPI_Comm comm )
    {
        _comm_ptr.reset(
            // Duplicate the communicator and store in a std::shared_ptr so that
            // all copies point to the same object
            [comm]()
            {
                auto p = std::make_unique<MPI_Comm>();
                MPI_Comm_dup( comm, p.get() );
                return p.release();
            }(),
            // Custom deleter to mark the communicator for deallocation
            []( MPI_Comm* p )
            {
                MPI_Comm_free( p );
                delete p;
            } );
    }

    /*!
      \brief Get the MPI communicator.
    */
    MPI_Comm comm() const { return *_comm_ptr; }

    /*!
      \brief Get the number of neighbor ranks that this rank will communicate
      with.
      \return The number of MPI ranks that will exchange data with this rank.
    */
    int numNeighbor() const { return _neighbors.size(); }

    /*!
      \brief Given a local neighbor id get its rank in the MPI communicator.
      \param neighbor The local id of the neighbor to get the rank for.
      \return The MPI rank of the neighbor with the given local id.
    */
    int neighborRank( const int neighbor ) const
    {
        return _neighbors[neighbor];
    }

    /*!
      \brief Get the number of elements this rank will export to a given
      neighbor.
      \param neighbor The local id of the neighbor to get the number of
      exports for.
      \return The number of elements this rank will export to the neighbor with
      the given local id.
     */
    std::size_t numExport( const int neighbor ) const
    {
        return _num_export[neighbor];
    }

    /*!
      \brief Get the total number of exports this rank will do.
      \return The total number of elements this rank will export to its
      neighbors.
    */
    std::size_t totalNumExport() const { return _total_num_export; }

    /*!
      \brief Get the number of elements this rank will import from a given
      neighbor.
      \param neighbor The local id of the neighbor to get the number of
      imports for.
      \return The number of elements this rank will import from the neighbor
      with the given local id.
     */
    std::size_t numImport( const int neighbor ) const
    {
        return _num_import[neighbor];
    }

    /*!
      \brief Get the total number of imports this rank will do.
      \return The total number of elements this rank will import from its
      neighhbors.
    */
    std::size_t totalNumImport() const { return _total_num_import; }

    /*!
      \brief Get the number of export elements.
      \return The number of export elements.

      Whenever the communication plan is applied, this is the total number of
      elements expected to be input on the sending ranks (in the forward
      communication plan). This will be different than the number returned by
      totalNumExport() if some of the export ranks used in the construction
      are -1 and therefore will not particpate in an export operation.
    */
    std::size_t exportSize() const { return _num_export_element; }

    /*!
      \brief Get the steering vector for the exports.
      \return The steering vector for the exports.

      The steering vector places exports in contiguous chunks by destination
      rank. The chunks are in consecutive order based on the local neighbor id
      (i.e. all elements going to neighbor with local id 0 first, then all
      elements going to neighbor with local id 1, etc.).
    */
    Kokkos::View<std::size_t*, memory_space> getExportSteering() const
    {
        return _export_steering;
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
        _num_export_element = element_export_ranks.size();

        // Store the unique neighbors (this rank first).
        _neighbors = getUniqueTopology( comm(), neighbor_ranks );
        int num_n = _neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Initialize import/export sizes.
        _num_export.assign( num_n, 0 );
        _num_import.assign( num_n, 0 );

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
            _num_export[n] = neighbor_counts_host( _neighbors[n] );

        // Post receives for the number of imports we will get.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &_num_import[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                           mpi_tag, comm(), &( requests.back() ) );
            }
            else
                _num_import[n] = _num_export[n];

        // Send the number of exports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
                MPI_Send( &_num_export[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                          mpi_tag, comm() );

        // Wait on receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Get the total number of imports/exports.
        _total_num_export =
            std::accumulate( _num_export.begin(), _num_export.end(), 0 );
        _total_num_import =
            std::accumulate( _num_import.begin(), _num_import.end(), 0 );

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( comm() );

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
        _num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( comm(), &my_rank );

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
        _neighbors.clear();
        _num_export.clear();
        _total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( neighbor_counts_host( r ) > 0 )
            {
                _neighbors.push_back( r );
                _num_export.push_back( neighbor_counts_host( r ) );
                _total_num_export += neighbor_counts_host( r );
                neighbor_counts_host( r ) = 1;
            }

        // Get the number of export ranks and initially allocate the import
        // sizes.
        int num_export_rank = _neighbors.size();
        _num_import.assign( num_export_rank, 0 );

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of imports to be the number of exports.
        bool self_send = false;
        for ( int n = 0; n < num_export_rank; ++n )
            if ( _neighbors[n] == my_rank )
            {
                std::swap( _neighbors[n], _neighbors[0] );
                std::swap( _num_export[n], _num_export[0] );
                // _num_import[0] = _num_export[0];
                self_send = true;
                break;
        
            }

        MPIX_Comm *xcomm;
        MPIX_Info *xinfo;
        MPIX_Comm_init(&xcomm, comm());
        MPIX_Info_init(&xinfo);

        int num_import_rank = -1;
        int *src;
        unsigned long* import_sizes;
        // std::vector<std::size_t> import_sizes( comm_size );
        MPIX_Alltoall_crs(num_export_rank, _neighbors.data(), 1, MPI_UNSIGNED_LONG, _num_export.data(),
                          &num_import_rank, &src, 1, MPI_UNSIGNED_LONG, (void**)&import_sizes,
                          xinfo, xcomm);
        
        MPIX_Info_free(&xinfo);
        MPIX_Comm_free(&xcomm);
        _total_num_import = std::accumulate(import_sizes, import_sizes + num_import_rank, 0UL);
            
        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Get the message source.
            const auto source = src[i];

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to.
            auto found_neighbor =
                std::find( _neighbors.begin(), _neighbors.end(), source );

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this.
            if ( found_neighbor == std::end( _neighbors ) )
            {
                _neighbors.push_back( source );
                _num_import.push_back( import_sizes[i] );
                _num_export.push_back( 0 );
            }

            // Otherwise if we already sent something to this neighbor that
            // means we already have a neighbor/export entry. Just assign the
            // import entry for that neighbor.
            else
            {
                auto n = std::distance( _neighbors.begin(), found_neighbor );
                _num_import[n] = import_sizes[i];
            }
        }

        MPIX_Free(src);
        MPIX_Free(import_sizes);

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( comm() );

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
        _neighbors = getUniqueTopology( comm(), neighbor_ranks );
        int num_n = _neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Initialize import/export sizes.
        _num_export.assign( num_n, 0 );
        _num_import.assign( num_n, 0 );

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
            _num_import[n] = neighbor_counts_host( _neighbors[n] );

        // Post receives to get the number of indices I will send to each rank.
        // Post that many wildcard recieves to get the number of indices I will
        // send to each rank
        std::vector<MPI_Request> requests;
        requests.reserve( num_n * 2 );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &_num_export[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                           mpi_tag, comm(), &( requests.back() ) );
            }
            else // Self import
            {
                _num_export[n] = _num_import[n];
            }

        // Send the number of imports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Isend( &_num_import[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                           mpi_tag, comm(), &( requests.back() ) );
            }

        // Wait on messages.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Get the total number of imports/exports.
        _total_num_export =
            std::accumulate( _num_export.begin(), _num_export.end(), 0 );
        _total_num_import =
            std::accumulate( _num_import.begin(), _num_import.end(), 0 );
        _num_export_element = _total_num_export;

        // Post receives to get the indices other processes are requesting
        // i.e. our export indices
        Kokkos::View<int*, memory_space> export_indices( "export_indices",
                                                         _total_num_export );
        size_t idx = 0;
        int num_messages = _total_num_export + element_import_ranks.extent( 0 );
        std::vector<MPI_Request> mpi_requests( num_messages );
        std::vector<MPI_Status> mpi_statuses( num_messages );
        for ( int i = 0; i < num_n; i++ )
        {
            for ( int j = 0; j < _num_export[i]; j++ )
            {
                MPI_Irecv( export_indices.data() + idx, 1, MPI_INT,
                           _neighbors[i], mpi_tag, comm(), &mpi_requests[idx] );
                idx++;
            }
        }

        // Send the indices we need
        for ( size_t i = 0; i < element_import_ranks.extent( 0 ); i++ )
        {
            MPI_Isend( element_import_ids.data() + i, 1, MPI_INT,
                       *( element_import_ranks.data() + i ), mpi_tag, comm(),
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
            "element_export_ranks_h", _total_num_export );
        for ( size_t i = 0; i < _total_num_export; i++ )
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
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int rank = -1;
        MPI_Comm_rank( comm(), &rank );

        _total_num_import = element_import_ranks.extent(0);

        // Step 1: Initialize indices
        Kokkos::View<int*, memory_space> indices("indices", _total_num_import);
        Kokkos::parallel_for("InitIndices", Kokkos::RangePolicy<ExecutionSpace>( 0, _total_num_import ), KOKKOS_LAMBDA(int i) {
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
        Kokkos::View<int*, memory_space> ranks_sorted("ranks_sorted", _total_num_import);
        Kokkos::View<int*, memory_space> ids_sorted("ids_sorted", _total_num_import);
        Kokkos::parallel_for("PermuteExports", Kokkos::RangePolicy<ExecutionSpace>( 0, _total_num_import ), KOKKOS_LAMBDA(int i) {
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
        _neighbors.clear();
        _num_export.clear();
        _num_import.clear();

        for ( size_t i = 0; i < neighbor_counts_host.extent( 0 ); i++ )
        {
            if ( neighbor_counts_host( i ) != 0 )
            {
                // Store we are importing this many from this rank
                _neighbors.push_back( i );
                _num_import.push_back( neighbor_counts_host( i ) );
            }
        }

        // Store offsets into ranks_sorted to send
        std::vector<int> sdispls;
        sdispls.push_back(0);
        for (int neighbor_rank : _neighbors)
            sdispls.push_back(sdispls.back() + neighbor_counts_host(neighbor_rank));
        
        // Store send counts to each rank
        std::vector<int> sendcounts;
        for (int neighbor_rank : _neighbors)
            sendcounts.push_back(neighbor_counts_host(neighbor_rank));


        // Assign all exports to zero
        _num_export.assign( _num_import.size(), 0 );


        MPIX_Comm *xcomm;
        MPIX_Info *xinfo;
        MPIX_Comm_init(&xcomm, comm());
        MPIX_Info_init(&xinfo);

        int num_export_rank = -1, total_num_export = -1;
        int *src, *recv_counts, *recv_displs, *recv_vals;
        // std::vector<std::size_t> import_sizes( comm_size );
        auto ids_sorted_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ids_sorted);
        MPIX_Alltoallv_crs(_neighbors.size(), _total_num_import, _neighbors.data(), sendcounts.data(), 
            sdispls.data(), MPI_INT, ids_sorted_host.data(),
            &num_export_rank, &total_num_export, &src, &recv_counts,
            &recv_displs, MPI_INT, (void**)&recv_vals, xinfo, xcomm); 
        
        
        _total_num_export = total_num_export;
        
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
            auto found = std::find(_neighbors.begin(), _neighbors.end(), exporting_rank);
            if (found == _neighbors.end())
            {
                _neighbors.push_back(exporting_rank);
                _num_import.push_back(0);
                _num_export.push_back(count);
            }
            else
            {
                int n = std::distance(_neighbors.begin(), found);
                _num_export[n] = count;
            }
        }

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of exports to be the number of imports.
        bool self_send = false;
        for ( int n = 0; n < _neighbors.size(); ++n )
            if ( _neighbors[n] == rank )
            {
                std::swap( _neighbors[n], _neighbors[0] );
                std::swap( _num_export[n], _num_export[0] );
                std::swap( _num_import[n], _num_import[0] );
                _num_export[0] = _num_import[0];
                self_send = true;
                break;
            }

        // Total number of imports and exports are now known
        _num_export_element = _total_num_export;

        // Now, build the export steering
        Kokkos::View<int*, Kokkos::HostSpace> element_export_ranks_h(
            "element_export_ranks", _total_num_export );
        Kokkos::View<int*, Kokkos::HostSpace> export_indices_h(
            "export_indices", _total_num_export );

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

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the send buffer of the communication plan. Ordered such that
      if a rank sends to itself then those values come first.

      \param neighbor_ids The id of each element in the neighbor send buffers.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.
    */
    template <class PackViewType, class RankViewType>
    void createExportSteering( const PackViewType& neighbor_ids,
                               const RankViewType& element_export_ranks )
    {
        // passing in element_export_ranks here as a dummy argument.
        createSteering( true, neighbor_ids, element_export_ranks,
                        element_export_ranks );
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the contiguous send buffer of the communication plan. Ordered
      such that if a rank sends to itself then those values come first.

      \param neighbor_ids The id of each element in the neighbor send buffers.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param element_export_ids The local ids of the elements to be
      exported. This corresponds with the export ranks vector and must be the
      same length if defined. The input is expected to be a Kokkos view or
      Cabana slice in the same memory space as the communication plan.
    */
    template <class PackViewType, class RankViewType, class IdViewType>
    void createExportSteering( const PackViewType& neighbor_ids,
                               const RankViewType& element_export_ranks,
                               const IdViewType& element_export_ids )
    {
        createSteering( false, neighbor_ids, element_export_ranks,
                        element_export_ids );
    }

    //! \cond Impl
    // Create the export steering vector.
    template <class ExecutionSpace, class PackViewType, class RankViewType,
              class IdViewType>
    void createSteering( ExecutionSpace, const bool use_iota,
                         const PackViewType& neighbor_ids,
                         const RankViewType& element_export_ranks,
                         const IdViewType& element_export_ids )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        if ( !use_iota &&
             ( element_export_ids.size() != element_export_ranks.size() ) )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( *_comm_ptr, &comm_size );

        // Calculate the steering offsets via exclusive prefix sum for the
        // exports.
        int num_n = _neighbors.size();
        std::vector<std::size_t> offsets( num_n, 0.0 );
        for ( int n = 1; n < num_n; ++n )
            offsets[n] = offsets[n - 1] + _num_export[n - 1];

        // Map the offsets to the device.
        Kokkos::View<std::size_t*, Kokkos::HostSpace> rank_offsets_host(
            Kokkos::ViewAllocateWithoutInitializing( "rank_map" ), comm_size );
        for ( int n = 0; n < num_n; ++n )
            rank_offsets_host( _neighbors[n] ) = offsets[n];
        auto rank_offsets = Kokkos::create_mirror_view_and_copy(
            memory_space(), rank_offsets_host );

        // Create the export steering vector for writing local elements into
        // the send buffer. Note we create a local, shallow copy - this is a
        // CUDA workaround for handling class private data.
        _export_steering = Kokkos::View<std::size_t*, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "export_steering" ),
            _total_num_export );
        auto steer_vec = _export_steering;
        Kokkos::parallel_for(
            "Cabana::createSteering",
            Kokkos::RangePolicy<ExecutionSpace>( 0, _num_export_element ),
            KOKKOS_LAMBDA( const int i ) {
                if ( element_export_ranks( i ) >= 0 )
                    steer_vec( rank_offsets( element_export_ranks( i ) ) +
                               neighbor_ids( i ) ) =
                        ( use_iota ) ? i : element_export_ids( i );
            } );
        Kokkos::fence();
    }

    template <class PackViewType, class RankViewType, class IdViewType>
    void createSteering( const bool use_iota, const PackViewType& neighbor_ids,
                         const RankViewType& element_export_ranks,
                         const IdViewType& element_export_ids )
    {
        // Use the default execution space.
        createSteering( execution_space{}, use_iota, neighbor_ids,
                        element_export_ranks, element_export_ids );
    }
    //! \endcond

  private:
    std::shared_ptr<MPI_Comm> _comm_ptr;
    std::vector<int> _neighbors;
    std::size_t _total_num_export;
    std::size_t _total_num_import;
    std::vector<std::size_t> _num_export;
    std::vector<std::size_t> _num_import;
    std::size_t _num_export_element;
    Kokkos::View<std::size_t*, memory_space> _export_steering;
};

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_MPIADVANCE_HPP
