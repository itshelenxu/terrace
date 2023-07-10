
/*
 * ============================================================================
 *
 *       Filename:  graph.h
 *
 *         Author:  Prashant Pandey (), ppandey@berkeley.edu
 *   Organization: Berkeley Lab
 *
 * ============================================================================
 */

#pragma once

#include <stdlib.h>

#include <set>
#include <unordered_set>
#include <vector>
#include <iostream>

#include <string.h>

#include "PMA.hpp"
// #include "partitioned_counter.h"
#include "btree.h"
// #include "BitArray.h"
//#include "cpp-btree/btree_set.h"

namespace graphstore {

#define PREFETCH 1
#define ENABLE_LOCK 0

#if WEIGHTED
#define NUM_IN_PLACE_NEIGHBORS 14
#else
#define NUM_IN_PLACE_NEIGHBORS 13
#endif
#define MEDIUM_DEGREE (1ULL << 10)

#define LOCK_MASK (1ULL << 31)
#define UNLOCK_MASK ~(1ULL << 31)

// A 0 bit means the lock is free
// A 1 bit means the lock is currently acquired
static inline void lock(uint32_t *data)
{
   while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 31)) != 0) {}
}

static inline void unlock(uint32_t *data)
{
   __sync_fetch_and_and(data, UNLOCK_MASK);
}

  // An efficient way to store graph topology for sparse (and skewed) DAGs.
  // Vertices are defined as 32-bit integers.
  // There can be only one edge between two vertices, i.e., simple graphs.
  // Each vertex and up to 13 neighbors are stored in place.
  // If there are more than 12 neighbors then they are stored in a
  // ordered container and a pointer is stored in place with the source
  // vertex.
  // Vertices are divided into three categories. First, low degree vertices --
  // degree smaller than 12. These vertices and neighbors are stored
  // completely in place.
  // Second, medium degree vertices -- degree between 13 and 100K. 12
  // neighbors are stored in place and rest of the neighbors are stored in the
  // second level container along with the neighbors of other medium degree
  // vertices.
  // Third, high degree vertices -- degree greater than 100K. 12
  // neighbors are stored in place and rest of the neighbors are stored in the
  // third level container. Each vertex gets a dedicated container.
  class TerraceGraph {
    public:
      typedef uint32_t vertex;
      typedef uint32_t weight;
#if WEIGHTED
      typedef std::pair<vertex, vertex> e;
      typedef std::pair<e, weight> edge;
#else
      typedef std::pair<vertex, vertex> edge;
#endif
      typedef PMA sl_container;
      typedef PMA::iterator sl_container_iterator;
      typedef BTree<vertex, weight> tl_container;
      typedef BTree<vertex, weight>::Iterator tl_container_iterator;

      // for backward compatibility. Will remove these typedefs in some time.
      typedef std::set<vertex> vertex_set;
      typedef std::set<edge> edge_set;
      typedef vertex_set::iterator vertex_set_iterator;

      class NeighborIterator {
        public:
          NeighborIterator(const TerraceGraph* graph, vertex v);
#if WEIGHTED
          std::pair<vertex, weight> operator*(void) const;
#else
          vertex operator*(void) const;
#endif
          void operator++(void);
          bool done(void) const;

        private:
          const TerraceGraph* g;
          vertex source;
          uint32_t degree;
          sl_container_iterator sl_it;
          sl_container_iterator sl_it_end;
          tl_container_iterator tl_it;
          uint32_t local_idx{0};
          vertex neighbors[NUM_IN_PLACE_NEIGHBORS]; // 4*13=52 Bytes
#if WEIGHTED
          weight weights[NUM_IN_PLACE_NEIGHBORS]; // 4*13=52 Bytes
#endif
      };

      TerraceGraph(uint32_t size); // create a graph with the given size (#num nodes)
      TerraceGraph(std::string prefix); // read graph from disk
      ~TerraceGraph();

      void serialize(std::string prefix); // write graph to disk

      // add edge between vertices s and d
      // if edge already exists will not do anything.
      // return -1 if the edge is not added for some reason.
#if WEIGHTED
      int add_edge(const vertex s, const vertex d, const weight w);
#else
      int add_edge(const vertex s, const vertex d);
#endif

			int remove_edge(const vertex s, const vertex d);

#if WEIGHTED
			void add_edge_batch(vertex *srcs,vertex *dests, weight *w, uint32_t
													edge_count, std::vector<uint32_t>& perm);
#else
			void add_edge_batch(vertex *srcs,vertex *dests, uint32_t edge_count,
													std::vector<uint32_t>& perm);
#endif
      
			void build_from_batch(vertex *srcs,vertex *dests, uint32_t vertex_count, uint32_t edge_count);


#if WEIGHTED
			void add_edge_batch_no_perm(vertex *srcs,vertex *dests, weight *w, uint32_t
													edge_count);
#else
			void add_edge_batch_no_perm(vertex *srcs,vertex *dests, uint32_t edge_count);
#endif

// TODO: FILLIN
/*
#if WEIGHTED
      int remove_edge_batch(const vertex *srcs, const vertex *dests, const weight *w, uint32_t edge_count);
#else
      int remove_edge_batch(const vertex *s, const vertex *d, uint32_t edge_count);
#endif
*/
      // check for the existence of the edge
      uint32_t is_edge(const vertex s, const vertex d);
      // get out neighbors of vertex s
      NeighborIterator neighbors(const vertex v) const;
      // get out degree of vertex v
      uint32_t degree(const vertex v) const;
      uint64_t get_size(void);
  
      template <class F, typename VS>
      void map_sparse(F &f, VS &output_vs, uint32_t self_index, bool output);
      //template <class F, typename VS>
      //void map_dense(F &f, VS &vs, uint32_t self_index, bool output);
      template <class F, typename VS>
      void map_dense_vs_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output);
      template <class F, typename VS>
      void map_dense_vs_not_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output);

      template <class F>
      void map_neighbors_no_early_exit(size_t i, F &&f) const;

      template <class F>
      void map_neighbors_early_exit(size_t i, F &&f) const;

      template <class F>
      void parallel_map_neighbors_no_early_exit(size_t i, F &&f) const;

      template <class F>
      void parallel_map_neighbors_early_exit(size_t i, F &&f) const;
      
      void verify_neighbors(size_t i, uint32_t* arr) const;

      uint64_t get_num_edges(void) const;
      uint32_t get_num_vertices(void) const;

      uint32_t count_common(const vertex s, const vertex d) const;
      void print_vertex_block(const vertex v) const;

    private:
#if WEIGHTED
	void add_inplace(vertex *srcs, vertex	*dests, weight *wghts,
			 						uint32_t i, std::vector<uint32_t>& parts, uint8_t
			 						*array_pma, uint8_t *array_btree, uint8_t *array_btree_node);
#else
	void add_inplace(vertex *srcs, vertex *dests, uint32_t i,
													std::vector<uint32_t>& parts, uint8_t *array_pma,
													uint8_t *array_btree, uint8_t *array_btree_node);
#endif
#if WEIGHTED
	void add_btree(vertex *srcs, vertex	*dests, const weight *wghts,
					 			uint32_t i, std::vector<uint32_t>& parts, uint8_t *array);
#else 
	void add_btree(vertex *srcs, vertex	*dests, uint32_t i,
											std::vector<uint32_t>& parts, uint8_t *array);
#endif

			inline bool is_btree(const vertex s) const {
				return vertices[s].aux_neighbors != nullptr;
			}
      bool check_in_place_dup(const vertex s, const vertex d) const;
      vertex get_pma_edge(uint64_t& idx) const;

      bool count_common_inplace(const vertex s, const vertex d, uint64_t&
                                s_idx, uint64_t& d_idx, uint32_t& count) const;
      bool count_common_inplace_pma(const vertex s, const vertex d, uint64_t&
                                    s_idx, uint64_t& d_idx, uint32_t& count)
        const;
      bool count_common_inplace_btree(const vertex s, const vertex d, uint64_t&
                                      s_idx, tl_container_iterator& d_it,
                                      uint32_t& count) const;
      bool count_common_pma_pma(const vertex s, const vertex d, uint64_t&
                                s_idx, uint64_t& d_idx, uint32_t& count) const;
      bool count_common_pma_btree(const vertex s, const vertex d, uint64_t&
                                  s_idx, tl_container_iterator& d_it,
                                  uint32_t& count) const;
      bool count_common_btree_btree(const vertex s, const vertex d,
                                    tl_container_iterator& s_it,
                                    tl_container_iterator& d_it, uint32_t&
                                    count) const;
      // Need to make this block cache line (64 Bytes) aligned. 
      typedef struct __attribute__ ((__packed__)) {
        uint32_t degree; // 4 Bytes
        vertex neighbors[NUM_IN_PLACE_NEIGHBORS]; // 4*13=52 Bytes
#if WEIGHTED
        weight weights[NUM_IN_PLACE_NEIGHBORS]; // 4*14*2=112 Bytes
        uint32_t padding; // 4 Bytes padding (112+4+4+8=128)
#endif
        void *aux_neighbors{nullptr};  // 8 Bytes
        } vertex_block;

      vertex_block *vertices;
      sl_container second_level;
      uint32_t num_vertices{0};
      uint64_t num_edges{0};
      uint64_t inplace_size{0};
  };

  inline TerraceGraph::TerraceGraph(uint32_t size) : second_level(size), num_vertices(size) {
    inplace_size = size * sizeof(vertex_block);
    vertices = (vertex_block*)calloc(size, sizeof(vertex_block));
  }

  inline TerraceGraph::~TerraceGraph() {
    free(vertices);
  }

  bool inline TerraceGraph::check_in_place_dup(const vertex s, const vertex d) const {
    for (uint32_t i = 0; i < NUM_IN_PLACE_NEIGHBORS; i++) {
      if (vertices[s].neighbors[i] == d)
        return true;
    }

    return false;
  }

#if WEIGHTED
	void inline TerraceGraph::add_inplace(vertex *srcs, vertex	*dests, weight *wghts,
													uint32_t idx, std::vector<uint32_t>& parts, uint8_t
													*array_pma, uint8_t *array_btree, uint8_t
													*array_btree_node) {
#else
	void inline TerraceGraph::add_inplace(vertex *srcs, vertex *dests, uint32_t idx,
													std::vector<uint32_t>& parts, uint8_t *array_pma,
													uint8_t *array_btree, uint8_t *array_btree_node) {
#endif

		vertex s = srcs[parts[idx]];
		std::pair<uint32_t, uint32_t> range = {parts[idx], parts[idx+1]};
		uint32_t size = range.second - range.first;

#if ENABLE_LOCK
		lock(&vertices[s].degree);
#endif
		uint32_t degree = this->degree(s);
		if (degree == 0) { // it's a new src. Copy neighbors to in place
			uint32_t cnt = size < NUM_IN_PLACE_NEIGHBORS ? size :
				NUM_IN_PLACE_NEIGHBORS;
			memcpy(vertices[s].neighbors, &dests[range.first], cnt*sizeof(vertex));
#if WEIGHTED
			memcpy(vertices[s].weights, &wghts[range.first], cnt*sizeof(weight));
#endif
			// mark appropriate bit vectors for rest of the neighbors
			if (vertices[s].aux_neighbors == nullptr && degree + size <=
					MEDIUM_DEGREE) { // going to pma
				for (uint32_t i = range.first + cnt; i < range.second; i++) {
					array_pma[i] = 1;
				}
			} else { // going to b-tree
				for (uint32_t i = range.first + cnt; i < range.second; i++) {
					array_btree[i] = 1;
				}
				array_btree_node[idx] = 1;
			}
			// update degree
			vertices[s].degree += cnt;
		} else { // some neighbors already exist
			// merge two sorted lists and find new in_place neighbors
			uint32_t in_place_limit = degree < NUM_IN_PLACE_NEIGHBORS ? degree :
				NUM_IN_PLACE_NEIGHBORS;
			vertex *new_in_place = (vertex*)calloc(degree + size, sizeof(vertex));
#if WEIGHTED
			weight *new_in_place_wghts = (weight*)calloc(degree + size, sizeof(vertex));
#endif
			uint32_t i{0}, j{0}, k{0};
			for (i = 0, j = range.first; i < in_place_limit && j < range.second;
					 k++) {
				if (vertices[s].neighbors[i] < dests[j]) {
					new_in_place[k] = vertices[s].neighbors[i];
#if WEIGHTED
					new_in_place_wghts[k] = vertices[s].weights[i];
#endif
					i++;
				} else if (vertices[s].neighbors[i] > dests[j]) {
					new_in_place[k] = dests[j];
#if WEIGHTED
					new_in_place_wghts[k] = wghts[j];
#endif
					j++;
				} else {
					new_in_place[k] = vertices[s].neighbors[i];
#if WEIGHTED
					new_in_place_wghts[k] = vertices[s].weights[i];
#endif
					i++; j++;
				}
			}
			if (i < in_place_limit) {
				memcpy(new_in_place + k, &vertices[s].neighbors[i],
							 (in_place_limit-i)*sizeof(vertex));
#if WEIGHTED
				memcpy(new_in_place_wghts + k, &vertices[s].weights[i],
							 (in_place_limit-i)*sizeof(weight));
#endif
				k += (in_place_limit-i);
			} else if (j < range.second) {
				memcpy(new_in_place + k, &dests[j], (range.second-j)*sizeof(vertex));
#if WEIGHTED
				memcpy(new_in_place_wghts + k, &wghts[j],
							 (range.second-j)*sizeof(weight));
#endif
				k += (range.second-j);
			}
			// copy new in place neighbors
			uint32_t cnt = k < NUM_IN_PLACE_NEIGHBORS ? k : NUM_IN_PLACE_NEIGHBORS;
			memcpy(vertices[s].neighbors, new_in_place, cnt*sizeof(vertex));
#if WEIGHTED
			memcpy(vertices[s].weights, new_in_place_wghts, cnt*sizeof(weight));
#endif
			// copy and mark appropriate bit vectors for rest of the neighbors
			if (k > NUM_IN_PLACE_NEIGHBORS) {
				uint32_t sec_cnt = k-NUM_IN_PLACE_NEIGHBORS;
				memcpy(dests+range.first, new_in_place + cnt, sec_cnt*sizeof(vertex));
#if WEIGHTED
				memcpy(wghts+range.first, new_in_place + cnt, sec_cnt*sizeof(weight));
#endif
				if (vertices[s].aux_neighbors == nullptr && degree + size <=
						MEDIUM_DEGREE) {
					for (uint32_t i = range.first; i < range.first + sec_cnt; i++) {
						array_pma[i] = 1;
					}
				} else {
					for (uint32_t i = range.first; i < range.first + sec_cnt; i++) {
						array_btree[i] = 1;
					}
					array_btree_node[idx] = 1;
				}
			}
			// update degree
			if (degree < NUM_IN_PLACE_NEIGHBORS) {
				vertices[s].degree = LOCK_MASK;
				vertices[s].degree = cnt;
			}
		}
#if ENABLE_LOCK
		unlock(&vertices[s].degree);
#endif
		return;
 	}

#if WEIGHTED
	void inline TerraceGraph::add_btree(vertex *srcs, vertex	*dests, const weight *wghts,
												uint32_t i, std::vector<uint32_t>& parts, uint8_t
												*array) {
#else 
	void inline TerraceGraph::add_btree(vertex *srcs, vertex	*dests, uint32_t i,
												std::vector<uint32_t>& parts, uint8_t *array) {
#endif
		vertex s = srcs[parts[i]];
		std::pair<uint32_t, uint32_t> range = {parts[i], parts[i+1]};

#if ENABLE_LOCK
		lock(&vertices[s].degree);
#endif
		if (vertices[s].aux_neighbors == nullptr) {
			tl_container *container = new tl_container();
			vertices[s].aux_neighbors = container;
			//TODO: the PMA might have more edges than MEDIUM_DEGREE.. Adding
			//twice the space might be safe
			uint32_t des[MEDIUM_DEGREE*2] = {0};
			uint32_t src[MEDIUM_DEGREE*2] = {0};
			uint32_t nedges = 0;
			// move neighbors from the second level
			auto end = second_level.end(s);
			for (auto it=second_level.begin(s); it!=end; ++it) {
#if WEIGHTED
				container->insert((*it).dest, (*it).value);
#else
				container->insert((*it).dest);
#endif
				des[nedges] = (*it).dest;
				src[nedges] = s;
				nedges++;
			}
			second_level.remove_edge_batch(src, des, nedges);
		}
		// insert the rest in the b-tree
		tl_container *container = (tl_container*)(vertices[s].aux_neighbors);
		for (uint32_t i = range.first; i < range.second; i++) {
			if (array[i] ==1) {
#if WEIGHTED
				if (container->insert(dests[i], wghts[i]))
					vertices[s].degree++;
#else
				if (container->insert(dests[i]))
					vertices[s].degree++;
#endif
			}
		}
#if ENABLE_LOCK
		unlock(&vertices[s].degree);
#endif
	}

// add edge in batch
#if WEIGHTED
	void inline TerraceGraph::add_edge_batch_no_perm(vertex *srcs, vertex *dests, weight *wghts,
														 uint32_t edge_count)
	{
#else
  void inline TerraceGraph::add_edge_batch_no_perm(vertex *srcs, vertex *dests, uint32_t edge_count) {
#endif
    printf("add edge batch no perm, edge count %u\n", edge_count);
		uint8_t *array_pma = (uint8_t*)calloc(edge_count, sizeof(uint8_t));
		uint8_t *array_btree = (uint8_t*)calloc(edge_count, sizeof(uint8_t));

		// generate partitions array
		std::vector<uint32_t> parts;
		vertex cur_src = srcs[0];
		parts.emplace_back(0);
		for (uint32_t i = 1; i < edge_count; i++) {
			if (cur_src != srcs[i]) {
				parts.emplace_back(i);
				cur_src = srcs[i];
			}
		}
		parts.emplace_back(edge_count);
		//BitArray array_btree_node(parts.size());
		uint8_t *array_btree_node = (uint8_t*)calloc(parts.size(), sizeof(uint8_t));

		// try and add edges in place and store overflow edges in sec_list
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    //uint32_t i = 0; i < parts.size()-1; i++) {
#if WEIGHTED
			add_inplace(srcs, dests, wghts, i, parts, array_pma, array_btree,
									array_btree_node);
#else
			add_inplace(srcs, dests, i, parts, array_pma, array_btree,
									array_btree_node);
#endif
		});
    printf("starting to insert items to pma\n");
		// insert edges from the sec list to PMA
		parlay::parallel_for (0, edge_count, [&](uint32_t i) { 
    // for (uint32_t i = 0; i < edge_count; i++) {
			auto idx = i;
			if (array_pma[idx] == 1) {
				vertex s = srcs[idx];
#if ENABLE_LOCK
				lock(&vertices[s].degree);
#endif
				uint32_t task_id = 1 + i*2;
#if WEIGHTED
				if (second_level.add_edge_update_fast(srcs[idx], dests[idx], wghts[idx], task_id))
					vertices[s].degree++;
#else
				if (second_level.add_edge_update_fast(srcs[idx], dests[idx], 1, task_id))
					vertices[s].degree++;
#endif
#if ENABLE_LOCK
				unlock(&vertices[s].degree);
#endif
			}
		});
    printf("done with pma\n");
		
		// insert edges from sec list to b-tree 
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    // for (uint32_t i = 0; i < parts.size()-1; i++) {
			if (array_btree_node[i] == 1) {
#if WEIGHTED
				add_btree(srcs, dests, wghts, i, parts, array_btree);
#else
				add_btree(srcs, dests, i, parts, array_btree);
#endif
			}
		});
	}

#if WEIGHTED
  int inline TerraceGraph::add_edge(const vertex s, const vertex d, const weight w) {
#else
  int inline TerraceGraph::add_edge(const vertex s, const vertex d) {
#endif
    uint32_t idx;
#if ENABLE_LOCK
    lock(&vertices[s].degree);
#endif
    uint32_t degree = this->degree(s);
    if (degree == 0) {
      vertices[s].neighbors[0] = d;
#if WEIGHTED
      vertices[s].weights[0] = w;
#endif
    } else {
#if PREFETCH
      if (!is_btree(s)) {
        __builtin_prefetch(vertices[s].aux_neighbors, 1);  
      } else if (degree > NUM_IN_PLACE_NEIGHBORS) {
        __builtin_prefetch(&second_level.nodes[s], 1);
      }
#endif
      vertex kicked = d;
#if WEIGHTED
      vertex kickedw = w;
#endif
			if (degree < NUM_IN_PLACE_NEIGHBORS ||
					d <= vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1]) {
				for (idx = 0; idx < degree && idx < NUM_IN_PLACE_NEIGHBORS; idx++) {
					if (vertices[s].neighbors[idx] < d)
						continue;
					else if (vertices[s].neighbors[idx] == d) {
#if WEIGHTED
						vertices[s].weights[idx] = w;
#endif
#if ENABLE_LOCK
						unlock(&vertices[s].degree);
#endif
						return -1;
					}
					else
						break;
				}
#if PREFETCH
				if (is_btree(s)) {
					__builtin_prefetch(((tl_container*)vertices[s].aux_neighbors)->get_root(),
														 1);  
				} else if (degree > NUM_IN_PLACE_NEIGHBORS) {
					__builtin_prefetch(&second_level.edges.dests[second_level.nodes[s].beginning], 1);
				}
#endif
				if (idx < NUM_IN_PLACE_NEIGHBORS) { // store in-place
					if (idx < degree) {
						if (degree >= NUM_IN_PLACE_NEIGHBORS) { // need to kick
							kicked = vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1];
#if WEIGHTED
							kickedw = vertices[s].weights[NUM_IN_PLACE_NEIGHBORS-1];
#endif
						}
						memmove(&vertices[s].neighbors[idx+1], &vertices[s].neighbors[idx],
										(NUM_IN_PLACE_NEIGHBORS-idx-1)*sizeof(vertices[s].neighbors[0]));
#if WEIGHTED
						memmove(&vertices[s].weights[idx+1], &vertices[s].weights[idx],
										(NUM_IN_PLACE_NEIGHBORS-idx-1)*sizeof(vertices[s].weights[0]));
#endif
					}
					vertices[s].neighbors[idx] = d;
#if WEIGHTED
					vertices[s].weights[idx] = w;
#endif
					if (degree < NUM_IN_PLACE_NEIGHBORS) {
						vertices[s].degree++;
#if ENABLE_LOCK
						unlock(&vertices[s].degree);
#endif
						return 0;
					}
				}
			}
      // if degree is strictly greater than MEDIUM_DEGREE directly insert in the
      // third level
      if (is_btree(s)) {
        tl_container *container = (tl_container*)(vertices[s].aux_neighbors);
#if WEIGHTED
        if (!container->insert(kicked, kickedw)) {
#else
        if (!container->insert(kicked)) {
#endif
#if ENABLE_LOCK
          unlock(&vertices[s].degree);
#endif
          return -1;
        }
        // check if insertion is in the second level or the third level
      } else {
#if WEIGHTED
        if (!second_level.add_edge_update(s, kicked, kickedw)) {
#else
        if (!second_level.add_edge_update(s, kicked, 1)) {
#endif
#if ENABLE_LOCK
          unlock(&vertices[s].degree);
#endif
          return -1;
        }
      }
    }
    vertices[s].degree++;

    degree = this->degree(s);
    // check to see if the second level needs to be moved to the third
    if (vertices[s].aux_neighbors == nullptr && degree > MEDIUM_DEGREE) {
      tl_container *container = new tl_container();
      vertices[s].aux_neighbors = container;
      uint32_t des[MEDIUM_DEGREE] = {0};
      uint32_t src[MEDIUM_DEGREE] = {0};
      uint32_t nedges = 0;
      // move neighbors from the second level
      auto end = second_level.end(s);
      for (auto it = second_level.begin(s); it != end; ++it) {
#if WEIGHTED
        container->insert((*it).dest, (*it).value);
#else
        container->insert((*it).dest);
#endif
        des[nedges] = (*it).dest;
        src[nedges] = s;
        nedges++;
      }
      second_level.remove_edge_batch(src, des, nedges);
    }

#if ENABLE_LOCK
    unlock(&vertices[s].degree);
#endif
    return 0;
  }

  uint32_t inline TerraceGraph::is_edge(const vertex s, const vertex d) {
    for (uint32_t i = 0; i < vertices[s].degree && i < NUM_IN_PLACE_NEIGHBORS;
         i++) {
      if (vertices[s].neighbors[i] == d)
#if WEIGHTED
        return vertices[s].weights[i];
#else
        return 1;
#endif
    }
    if (vertices[s].degree > NUM_IN_PLACE_NEIGHBORS) {
      if (!is_btree(s)) {
        return second_level.find_value(s, d);
      } else {
#if WEIGHTED
        return ((tl_container*)(vertices[s].aux_neighbors))->get_val(d);
#else
        auto const  it = ((tl_container*)(vertices[s].aux_neighbors))->find(d);
        if (it != nullptr)
          return 1;
#endif
      }
    }
    //auto it = NeighborIterator(this, s);
    //while (!it.done()) {
    //if (*it == d)
    //return true;
    //++it;
    //}
    return 0;
  }


	void inline TerraceGraph::build_from_batch(vertex *srcs, vertex *dests, uint32_t vertex_count, uint32_t edge_count) {
		uint8_t *array_pma = (uint8_t*)calloc(edge_count, sizeof(uint8_t));
		uint8_t *array_btree = (uint8_t*)calloc(edge_count, sizeof(uint8_t));

		// generate partitions array
		std::vector<uint32_t> parts;
		vertex cur_src = srcs[0];
		parts.emplace_back(0);
		for (uint32_t i = 1; i < edge_count; i++) {
			if (cur_src != srcs[i]) {
				parts.emplace_back(i);
				cur_src = srcs[i];
			}
		}
		parts.emplace_back(edge_count);
		//BitArray array_btree_node(parts.size());
		uint8_t *array_btree_node = (uint8_t*)calloc(parts.size(), sizeof(uint8_t));

		// try and add edges in place and store overflow edges in sec_list
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    //uint32_t i = 0; i < parts.size()-1; i++) {
#if WEIGHTED
			add_inplace(srcs, dests, wghts, i, parts, array_pma, array_btree,
									array_btree_node);
#else
			add_inplace(srcs, dests, i, parts, array_pma, array_btree,
									array_btree_node);
#endif
		});

    uint32_t edges_for_pma = 0;
    uint32_t edges_for_btree = 0;
    for(uint32_t i = 0; i < edge_count; i++) {
      edges_for_pma += array_pma[i];
      edges_for_btree += array_btree[i];
    }
    printf("starting to insert items to pma\n");
    printf("edges for pma %u, edges for btree %u\n", edges_for_pma, edges_for_btree);
		// insert edges from the sec list to PMA
		uint32_t *additional_degrees = (uint32_t*)calloc(vertex_count, sizeof(uint32_t));

    // populate additional_degrees with how many went to each node in the PMA
    second_level.build_from_edges(srcs, dests, array_pma, vertex_count, edge_count, additional_degrees);

    for (uint32_t i = 0; i < vertex_count; i++) {
      vertices[i].degree += additional_degrees[i]; 
    }
    printf("done with pma\n");
		
		// insert edges from sec list to b-tree 
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    // for (uint32_t i = 0; i < parts.size()-1; i++) {
			if (array_btree_node[i] == 1) {
#if WEIGHTED
				add_btree(srcs, dests, wghts, i, parts, array_btree);
#else
				add_btree(srcs, dests, i, parts, array_btree);
#endif
			}
		});
	  for (uint32_t i = 0; i < num_vertices; i++)
			num_edges += vertices[i].degree;
    printf("total num edges after build = %lu\n");
  }

// add edge in batch
#if WEIGHTED
	void inline TerraceGraph::add_edge_batch(vertex *srcs, vertex *dests, weight *wghts,
														 uint32_t edge_count, std::vector<uint32_t>& perm)
	{
#else
	void inline TerraceGraph::add_edge_batch(vertex *srcs, vertex *dests, uint32_t edge_count,
														 std::vector<uint32_t>& perm) {
#endif
    printf("starting add edge batch \n");
		uint8_t *array_pma = (uint8_t*)calloc(edge_count, sizeof(uint8_t));
		uint8_t *array_btree = (uint8_t*)calloc(edge_count, sizeof(uint8_t));

		// generate partitions array
		std::vector<uint32_t> parts;
		vertex cur_src = srcs[0];
		parts.emplace_back(0);
		for (uint32_t i = 1; i < edge_count; i++) {
			if (cur_src != srcs[i]) {
				parts.emplace_back(i);
				cur_src = srcs[i];
			}
		}
		parts.emplace_back(edge_count);
		//BitArray array_btree_node(parts.size());
		uint8_t *array_btree_node = (uint8_t*)calloc(parts.size(), sizeof(uint8_t));

		// try and add edges in place and store overflow edges in sec_list
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    //uint32_t i = 0; i < parts.size()-1; i++) {
#if WEIGHTED
			add_inplace(srcs, dests, wghts, i, parts, array_pma, array_btree,
									array_btree_node);
#else
			add_inplace(srcs, dests, i, parts, array_pma, array_btree,
									array_btree_node);
#endif
		});

    uint32_t edges_for_pma = 0;
    uint32_t edges_for_btree = 0;
    for(uint32_t i = 0; i < edge_count; i++) {
      edges_for_pma += array_pma[i];
      edges_for_btree += array_btree[i];
    }
    printf("starting to insert items to pma\n");
    printf("edges for pma %u, edges for btree %u\n", edges_for_pma, edges_for_btree);
		// insert edges from the sec list to PMA
		// parlay::parallel_for (0, edge_count, [&](uint32_t i) { 
    for (uint32_t i = 0; i < edge_count; i++) {
			auto idx = perm[i];
			if (array_pma[idx] == 1) {
				vertex s = srcs[idx];
#if ENABLE_LOCK
				lock(&vertices[s].degree);
#endif
				uint32_t task_id = 1 + i*2;
#if WEIGHTED
				if (second_level.add_edge_update_fast(srcs[idx], dests[idx], wghts[idx], task_id))
					vertices[s].degree++;
#else
				if (second_level.add_edge_update_fast(srcs[idx], dests[idx], 1, task_id))
					vertices[s].degree++;
#endif
#if ENABLE_LOCK
				unlock(&vertices[s].degree);
#endif
			}
    }
		// });
    printf("done with pma\n");
		
		// insert edges from sec list to b-tree 
		parlay::parallel_for (0, parts.size() - 1, [&](uint32_t i) { 
    // for (uint32_t i = 0; i < parts.size()-1; i++) {
			if (array_btree_node[i] == 1) {
#if WEIGHTED
				add_btree(srcs, dests, wghts, i, parts, array_btree);
#else
				add_btree(srcs, dests, i, parts, array_btree);
#endif
			}
		});
	}

	int inline TerraceGraph::remove_edge(const vertex s, const vertex d) {
		uint32_t idx;
		int ret{0};
#if ENABLE_LOCK
		lock(&vertices[s].degree);
#endif
		uint32_t degree = this->degree(s);

		if (degree == 0) {
			ret = -1;
			goto unlock;
		}
		else {
#if PREFETCH
			if (is_btree(s)) {
				__builtin_prefetch(vertices[s].aux_neighbors, 1);  
			} else {
				__builtin_prefetch(&second_level.nodes[s], 1);
			}
#endif
			if (degree <= NUM_IN_PLACE_NEIGHBORS) { // only in place
				for (idx = 0; idx < degree; idx++) {
					if (vertices[s].neighbors[idx] < d)
						continue;
					else if (vertices[s].neighbors[idx] == d) {
						memcpy(&vertices[s].neighbors[idx], &vertices[s].neighbors[idx+1],
									 (degree-idx-1)*sizeof(vertex));
#if WEIGHTED
						memcpy(&vertices[s].weights[idx], &vertices[s].weights[idx+1],
									 (degree-idx-1)*sizeof(weight));
#endif
						goto decr;
					}
					else
						break;
				}
				if (idx == degree) {
					ret = -1;
					goto unlock;
				}
			} else if (d > vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1]) {
				// only secondary
				if (is_btree(s)) {
					tl_container *container = (tl_container*)(vertices[s].aux_neighbors);
					if (!container->remove(d)) {
						ret = -1;
						goto unlock;
					}
				} else {
					if (!second_level.remove_edge(s, d)) {
						ret = -1;
						goto unlock;
					}
				}
			} else { // both in place and secondary
				// first remove from in place
				for (idx = 0; idx < NUM_IN_PLACE_NEIGHBORS; idx++) {
					if (vertices[s].neighbors[idx] < d)
						continue;
					else if (vertices[s].neighbors[idx] == d) {
						memcpy(&vertices[s].neighbors[idx], &vertices[s].neighbors[idx+1],
									 (NUM_IN_PLACE_NEIGHBORS-idx-1)*sizeof(vertex));
#if WEIGHTED
						memcpy(&vertices[s].weights[idx], &vertices[s].weights[idx+1],
									 (NUM_IN_PLACE_NEIGHBORS-idx-1)*sizeof(weight));
#endif
						break;
					}
					else
						break;
				}
				// remove the smallest from sec
				vertex bump{0};
#if WEIGHTED
				weight bumpw{0};
#endif
				if (is_btree(s)) {
					tl_container *container = (tl_container*)(vertices[s].aux_neighbors);
					auto it = container->begin();
#if WEIGHTED
					bump = (*it).first;
					bumpw = (*it).second;
#else
					bump = *it;
#endif
					if (!container->remove(bump)) {
						ret = -1;
						goto unlock;
					}
				} else {
					auto it = second_level.begin(s);
					bump = (*it).dest;
#if WEIGHTED
					bumpw = (*it).value;
#endif
					if (!second_level.remove_edge(s, bump)) {
						ret = -1;
						goto unlock;
					}
				}
				// insert in in place
				vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1] = bump;	
#if WEIGHTED
				vertices[s].weights[NUM_IN_PLACE_NEIGHBORS-1] = bumpw;
#endif
			}
		}

decr:
		vertices[s].degree--;
unlock:
#if ENABLE_LOCK
		unlock(&vertices[s].degree);
#endif

		return ret;
	}

  inline TerraceGraph::NeighborIterator TerraceGraph::neighbors(const vertex v) const {
    return NeighborIterator(this, v);
  }

  inline uint32_t TerraceGraph::degree(const vertex v) const {
#if ENABLE_LOCK
    return vertices[v].degree & UNLOCK_MASK;
#else
    return vertices[v].degree;
#endif
  }

  uint64_t inline TerraceGraph::get_size(void) {
    uint64_t total_size{0};
    total_size += inplace_size;
    total_size += second_level.get_size();
    for (uint32_t idx = 0; idx < num_vertices; idx++) {
      if (vertices[idx].aux_neighbors != nullptr)
        total_size +=
          ((tl_container*)(vertices[idx].aux_neighbors))->get_size(); 
    }
    return total_size;
  }

  uint64_t inline TerraceGraph::get_num_edges(void) const {
    /*
		uint64_t num_edges{0};
		for (uint32_t i = 0; i < num_vertices; i++)
			num_edges += vertices[i].degree;
    */
		return num_edges;
  }

  uint32_t inline TerraceGraph::get_num_vertices(void) const {
    return num_vertices;
  }

  void inline TerraceGraph::print_vertex_block(const vertex v) const {
    std::cout << "Vertex: " << v << "\n";
    std::cout << "Degree: " << vertices[v].degree << "\n";
    std::cout << "In place neighbors: \n";
    for (uint32_t i = 0; i < vertices[v].degree && i < NUM_IN_PLACE_NEIGHBORS;
         i++) {
      std::cout << vertices[v].neighbors[i] << ", "
#if WEIGHTED
        <<  vertices[v].weights[i]<< ", "
#endif
        ;
    }
    std::cout << '\b';
    std::cout << '\b' << '\n';
  }

  inline TerraceGraph::NeighborIterator::NeighborIterator(const TerraceGraph *graph, TerraceGraph::vertex
                                            v) : g(graph), source(v),
  degree(g->vertices[source].degree) {
    memcpy(neighbors, g->vertices[source].neighbors,
           NUM_IN_PLACE_NEIGHBORS*sizeof(vertex));
#if WEIGHTED
    memcpy(weights, g->vertices[source].weights,
           NUM_IN_PLACE_NEIGHBORS*sizeof(weight));
#endif
    if (g->is_btree(source)) {
      tl_it = ((tl_container*)(g->vertices[source].aux_neighbors))->begin();
    }
    else {
      sl_it = graph->second_level.begin(v);
      sl_it_end = graph->second_level.end(v);
    }
  }

#if WEIGHTED
  std::pair<TerraceGraph::vertex, TerraceGraph::weight>
    TerraceGraph::NeighborIterator::operator*(void) const {
#else
  TerraceGraph::vertex inline TerraceGraph::NeighborIterator::operator*(void) const {
#endif
    if (local_idx < NUM_IN_PLACE_NEIGHBORS)
#if WEIGHTED
      return std::make_pair(neighbors[local_idx], weights[local_idx]);
#else
      return neighbors[local_idx];
#endif
    else if (!g->is_btree(source))
#if WEIGHTED
      return std::make_pair((*sl_it).dest, (*sl_it).value);
#else
      return (*sl_it).dest;
#endif
    else
      return *tl_it;
  }

  void inline TerraceGraph::NeighborIterator::operator++(void) {
    if (local_idx < NUM_IN_PLACE_NEIGHBORS && local_idx < degree)
      local_idx++;
    else if (!g->is_btree(source))
      ++sl_it;
    else
      ++tl_it;
  }

  bool inline TerraceGraph::NeighborIterator::done(void) const {
    if (local_idx == degree)
      return true;
    else if (local_idx == NUM_IN_PLACE_NEIGHBORS)
      return !g->is_btree(source) ? sl_it == sl_it_end : tl_it.done();
    return false;
  }

  bool inline TerraceGraph::count_common_inplace(const vertex s, const vertex d, uint64_t&
                                   s_idx, uint64_t& d_idx, uint32_t& count)
    const {
      uint64_t s_end = vertices[s].degree < NUM_IN_PLACE_NEIGHBORS ?
        vertices[s].degree : NUM_IN_PLACE_NEIGHBORS;
      uint64_t d_end = vertices[d].degree < NUM_IN_PLACE_NEIGHBORS ?
        vertices[d].degree : NUM_IN_PLACE_NEIGHBORS;

      auto s_v = vertices[s].neighbors[s_idx];
      auto d_v = vertices[d].neighbors[d_idx];
      while (s_idx < s_end && d_idx < d_end && s_v < s && d_v < d) {
        if (s_v == d_v) {
          count++; s_idx++; d_idx++;
        } else {
          s_idx += (s_v < d_v);
          d_idx += (d_v < s_v);
        }
        if (s_idx < s_end) s_v = vertices[s].neighbors[s_idx];
        if (d_idx < d_end) d_v = vertices[d].neighbors[d_idx];
      }
      return (s_v < s && d_v < d);
  }

  inline TerraceGraph::vertex TerraceGraph::get_pma_edge(uint64_t& idx) const {
    auto v = second_level.edges.dests[idx];
    if (v == NULL_VAL) {
      idx = ((idx >> second_level.edges.loglogN) +1 ) <<
        (second_level.edges.loglogN);
      v = second_level.edges.dests[idx];
    }
    return v;
  }

  bool inline TerraceGraph::count_common_inplace_pma(const vertex s, const vertex d,
                                       uint64_t& s_idx, uint64_t& d_idx,
                                       uint32_t& count) const {
    uint64_t s_end = vertices[s].degree < NUM_IN_PLACE_NEIGHBORS ?
      vertices[s].degree : NUM_IN_PLACE_NEIGHBORS;
    uint64_t d_end = second_level.nodes[d].end;

    auto s_v = vertices[s].neighbors[s_idx];
    auto d_v = get_pma_edge(d_idx);
    while (s_idx < s_end && d_idx < d_end && s_v < s && d_v < d) {
      if (d_v != NULL_VAL) {
        if (s_v == d_v) {
          count++; s_idx++; d_idx++;
        } else {
          s_idx += (s_v < d_v);
          d_idx += (d_v < s_v);
        }
      } else {
        d_idx = ((d_idx >> second_level.edges.loglogN) +1 ) <<
          (second_level.edges.loglogN);
      }
      if (s_idx < s_end) s_v = vertices[s].neighbors[s_idx];
      if (d_idx < d_end) d_v = get_pma_edge(d_idx);
    }
    return (s_v < s && d_v < d);
  }
  bool inline TerraceGraph::count_common_inplace_btree(const vertex s, const vertex d,
                                         uint64_t& s_idx,
                                         tl_container_iterator& d_it,
                                         uint32_t& count) const {
    uint64_t s_end = vertices[s].degree < NUM_IN_PLACE_NEIGHBORS ?
      vertices[s].degree : NUM_IN_PLACE_NEIGHBORS;

    auto s_v = vertices[s].neighbors[s_idx];
#if WEIGHTED
    auto d_v = (*d_it).first;
#else
    auto d_v = *d_it;
#endif
    while (s_idx < s_end && !d_it.done() && s_v < s && d_v < d) {
      if (s_v == d_v) {
        count++; s_idx++; ++d_it;
      } else {
        s_idx += (s_v < d_v);
        if (d_v < s_v) ++d_it;
      }
      if (s_idx < s_end) s_v = vertices[s].neighbors[s_idx];
#if WEIGHTED
      d_v = (*d_it).first;
#else
      d_v = *d_it;
#endif
    }
    return (s_v < s && d_v < d);
  }
  bool inline TerraceGraph::count_common_pma_pma(const vertex s, const vertex d, uint64_t&
                                   s_idx, uint64_t& d_idx, uint32_t& count)
    const {
      uint64_t s_end = second_level.nodes[s].end;
      uint64_t d_end = second_level.nodes[d].end;

      auto s_v = get_pma_edge(s_idx);
      auto d_v = get_pma_edge(d_idx);
      while (s_idx < s_end && d_idx < d_end && s_v < s && d_v < d) {
        if (s_v != NULL_VAL && d_v != NULL_VAL) {
          if (s_v == d_v) {
            count++; s_idx++; d_idx++;
          } else {
            s_idx += (s_v < d_v);
            d_idx += (d_v < s_v);
          }
        } else {
          if (s_v == NULL_VAL)
            s_idx = ((s_idx >> second_level.edges.loglogN) +1 ) <<
              (second_level.edges.loglogN);
          if (d_v == NULL_VAL)
            d_idx = ((d_idx >> second_level.edges.loglogN) +1 ) <<
              (second_level.edges.loglogN);
        }
        if (s_idx < s_end) s_v = get_pma_edge(s_idx);
        if (d_idx < d_end) d_v = get_pma_edge(d_idx);
      }
      return (s_v < s && d_v < d);
  }
  bool inline TerraceGraph::count_common_pma_btree(const vertex s, const vertex d, uint64_t&
                                     s_idx, tl_container_iterator& d_it,
                                     uint32_t& count) const {
    uint64_t s_end = second_level.nodes[s].end;

    auto s_v = get_pma_edge(s_idx);
#if WEIGHTED
    auto d_v = (*d_it).first;
#else
    auto d_v = *d_it;
#endif
    while (s_idx < s_end && !d_it.done() && s_v < s && d_v < d) {
      if (s_v != NULL_VAL) {
        if (s_v == d_v) {
          count++; s_idx++; ++d_it;
        } else {
          s_idx += (s_v < d_v);
          if (d_v < s_v) ++d_it;
        }
      } else {
        s_idx = ((s_idx >> second_level.edges.loglogN) +1 ) <<
          (second_level.edges.loglogN);
      }
      if (s_idx < s_end) s_v = get_pma_edge(s_idx);
#if WEIGHTED
      d_v = (*d_it).first;
#else
      d_v = *d_it;
#endif
    }
    return (s_v < s && d_v < d);
  }
  bool inline TerraceGraph::count_common_btree_btree(const vertex s, const vertex d,
                                       tl_container_iterator& s_it,
                                       tl_container_iterator& d_it,
                                       uint32_t& count) const {
#if WEIGHTED
    auto s_v = (*s_it).first;
#else
    auto s_v = *s_it;
#endif
#if WEIGHTED
    auto d_v = (*d_it).first;
#else
    auto d_v = *d_it;
#endif
    while (!s_it.done() && !d_it.done() && s_v < s && d_v < d) {
      if (s_v == d_v) {
        count++; ++s_it; ++d_it;
      } else {
        if (s_v < d_v) ++s_it;
        if (d_v < s_v) ++d_it;
      }
#if WEIGHTED
      s_v = (*s_it).first;
#else
      s_v = *s_it;
#endif
#if WEIGHTED
      d_v = (*d_it).first;
#else
      d_v = *d_it;
#endif
    }
    return (s_v < s && d_v < d);
  }

  uint32_t inline TerraceGraph::count_common(const vertex s, const vertex d) const {
    uint32_t final_count{0}, local_count{0};
    uint64_t s_idx(0), d_idx{0};
    tl_container_iterator s_it, d_it;
    bool ret{true};

#if PREFETCH
    if (is_btree(s)) {
      __builtin_prefetch(vertices[s].aux_neighbors);  
    } else if (vertices[s].degree > NUM_IN_PLACE_NEIGHBORS) {
      __builtin_prefetch(&second_level.nodes[s]);  
    }
    if (is_btree(d)) {
      __builtin_prefetch(vertices[d].aux_neighbors);  
    } else if (vertices[d].degree > NUM_IN_PLACE_NEIGHBORS) {
      __builtin_prefetch(&second_level.nodes[d]);  
    }
#endif

    ret = count_common_inplace(s, d, s_idx, d_idx, local_count);
    final_count += local_count;
    if (!ret)
      return final_count;

#if PREFETCH
    if (is_btree(s)) {
      __builtin_prefetch(((tl_container*)vertices[s].aux_neighbors)->get_root());  
    } else if (vertices[s].degree > NUM_IN_PLACE_NEIGHBORS) {
      __builtin_prefetch(&second_level.edges.dests[second_level.nodes[s].beginning]);
    }
    if (is_btree(d)) {
      __builtin_prefetch(((tl_container*)vertices[d].aux_neighbors)->get_root());  
    } else if (vertices[d].degree > NUM_IN_PLACE_NEIGHBORS) {
      __builtin_prefetch(&second_level.edges.dests[second_level.nodes[d].beginning]);
    }
#endif

    auto s_degree = vertices[s].degree;
    auto d_degree = vertices[d].degree;
    bool s_next{false}, d_next{false};
    if (s_idx == NUM_IN_PLACE_NEIGHBORS) {
      if (!is_btree(s))
        s_idx = second_level.nodes[s].beginning + 1;
      else
        s_it =  ((tl_container*)(vertices[s].aux_neighbors))->begin();
      s_next = true;
    }
    if (d_idx == NUM_IN_PLACE_NEIGHBORS) {
      if (!is_btree(d))
        d_idx = second_level.nodes[d].beginning + 1;
      else
        d_it =  ((tl_container*)(vertices[d].aux_neighbors))->begin();
      d_next = true;
    }

    local_count = 0;
    // both move to the next level
    if (s_next && d_next) {
      if (!is_btree(s) && !is_btree(d)) {
        ret = count_common_pma_pma(s, d, s_idx, d_idx, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } else if (!is_btree(s) && is_btree(d)) {
        ret = count_common_pma_btree(s, d, s_idx, d_it, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } else if (is_btree(s) && !is_btree(d)) {
        ret = count_common_pma_btree(d, s, d_idx, s_it, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } else {
        ret = count_common_btree_btree(s, d, s_it, d_it, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      }
    } else if (!s_next) { // one still in in-place
      if (!is_btree(d)) {
        ret = count_common_inplace_pma(s, d, s_idx, d_idx, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } else {
        ret = count_common_inplace_btree(s, d, s_idx, d_it, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } // either s/d is done or s need to move to the next level
      local_count = 0;
      if (s_idx == NUM_IN_PLACE_NEIGHBORS && s_degree >
          NUM_IN_PLACE_NEIGHBORS) { // s moved to the next level
        if (!is_btree(s)) {
          s_idx = second_level.nodes[s].beginning + 1;
          if (!is_btree(d)) {
            ret = count_common_pma_pma(s, d, s_idx, d_idx, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          } else {
            ret = count_common_pma_btree(s, d, s_idx, d_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          }
        } else {
          s_it =  ((tl_container*)(vertices[s].aux_neighbors))->begin();
          if (!is_btree(d)) {
            ret = count_common_pma_btree(d, s, d_idx, s_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          } else {
            ret = count_common_btree_btree(s, d, s_it, d_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          }
        }
      }
    } else if (!d_next) { // one still in in-place
      if (!is_btree(s)) {
        ret = count_common_inplace_pma(d, s, d_idx, s_idx, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } else {
        ret = count_common_inplace_btree(d, s, d_idx, s_it, local_count);
        final_count += local_count;
        if (!ret)
          return final_count;
      } // either s/d is done or s need to move to the next level
      local_count = 0;
      if (d_idx == NUM_IN_PLACE_NEIGHBORS && d_degree >
          NUM_IN_PLACE_NEIGHBORS) { // s moved to the next level
        if (!is_btree(d)) {
          d_idx = second_level.nodes[d].beginning + 1;
          if (!is_btree(s)) {
            ret = count_common_pma_pma(d, s, d_idx, s_idx, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          } else {
            ret = count_common_pma_btree(d, s, d_idx, s_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          }
        } else {
          d_it =  ((tl_container*)(vertices[d].aux_neighbors))->begin();
          if (!is_btree(s)) {
            ret = count_common_pma_btree(s, d, s_idx, d_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          } else {
            ret = count_common_btree_btree(d, s, d_it, s_it, local_count);
            final_count += local_count;
            if (!ret)
              return final_count;
          }
        }
      }
    } else {
      std::cerr << "This case is not possible while count common\n";
      abort();
    }
    return final_count;
  }

	template<class F, typename VS> struct Btree_map {
		VS &vs;
		F f;
		bool output;
		TerraceGraph::vertex self_index;
		TerraceGraph::tl_container* btree;

		Btree_map(VS &vs, F &f, bool output, TerraceGraph::vertex self_index,
							TerraceGraph::tl_container* btree) : vs(vs),
		f(f), output(output), self_index(self_index), btree(btree) {}

#if WEIGHTED
		void update(TerraceGraph::vertex v, TerraceGraph::weight w) {
			if (f.cond(v) == 1 && f.updateAtomic(self_index, v, w) == 1) {
				if (output) {
					vs.insert_sparse(v);
				}
			}
		}
#else
		void update(TerraceGraph::vertex v) {
			if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
				if (output) {
					vs.insert_sparse(v);
				}
			}
		}
#endif
	};

  template <class F>
  void inline TerraceGraph::map_neighbors_no_early_exit(size_t i, F &&f) const {
      uint32_t degree = vertices[i].degree;
      uint32_t local_idx = 0;
      // printf("map no early exit for vtx %lu, degree %u\n", i, degree);
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          f(i, v);
          ++local_idx;  
        }
      } else { //degree > num_in_place
#if PREFETCH
        if (is_btree(i)) {
          __builtin_prefetch(vertices[i].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[i]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          f(i, v);
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(i)) {
              __builtin_prefetch(((tl_container*)vertices[i].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[i].beginning]);
            }
          }
#endif
        }
        if (!is_btree(i)) {
          uint64_t idx = second_level.nodes[i].beginning + 1;
          uint64_t idx_end = second_level.nodes[i].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
#endif
              f(i, v);
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          auto it = ((tl_container*)(vertices[i].aux_neighbors))->begin();
          while (!it.done()) {
#if WEIGHTED
            auto v = (*it).first;
            auto w = (*it).second;
#else
            auto v = *it;
            f(i, v);
#endif
            ++it;
				}
      }
    }
  }

  // parallel map
  template <class F>
  void TerraceGraph::parallel_map_neighbors_no_early_exit(size_t i, F &&f) const {
      uint32_t degree = vertices[i].degree;
      uint32_t local_idx = 0;
      // just keep the small-degree map serial
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          f(i, v);
          ++local_idx;  
        }
      } else { //degree > num_in_place
#if PREFETCH
        if (is_btree(i)) {
          __builtin_prefetch(vertices[i].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[i]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          f(i, v);
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(i)) {
              __builtin_prefetch(((tl_container*)vertices[i].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[i].beginning]);
            }
          }
#endif
        }
        if (!is_btree(i)) {
          uint64_t idx = second_level.nodes[i].beginning + 1;
          uint64_t idx_end = second_level.nodes[i].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
#endif
              f(i, v);
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          ((tl_container*)(vertices[i].aux_neighbors))->parallel_map(i, f);
      }
    }
  }

  // verify neighbors
  void inline TerraceGraph::verify_neighbors(size_t i, uint32_t* arr) const {
      uint32_t degree = vertices[i].degree;
      uint32_t local_idx = 0;
      uint32_t edges_so_far = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[i].neighbors[local_idx];
          if (v != arr[edges_so_far]) {
            printf("IN PLACE: vtx %lu, position %u, got ngh %u, should be %u\n", i, edges_so_far, v, arr[edges_so_far]);
          }
          ++edges_so_far;
          ++local_idx;  
        }
      } else { //degree > num_in_place
#if PREFETCH
        if (is_btree(i)) {
          __builtin_prefetch(vertices[i].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[i]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[i].neighbors[local_idx];
          if (v != arr[edges_so_far]) {
            printf("IN PLACE: vtx %lu, position %u, got ngh %u, should be %u\n", i, edges_so_far, v, arr[edges_so_far]);  
          }

#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          ++local_idx;
          ++edges_so_far;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(i)) {
              __builtin_prefetch(((tl_container*)vertices[i].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[i].beginning]);
            }
          }
#endif
        }
        if (!is_btree(i)) {
          uint64_t idx = second_level.nodes[i].beginning + 1;
          uint64_t idx_end = second_level.nodes[i].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
              if (v != arr[edges_so_far]) {
                printf("\tIN PMA LEVEL: vtx %lu, position %u, got ngh %u, should be %u\n", i, edges_so_far, v, arr[edges_so_far]);
              }
              ++edges_so_far;
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          auto it = ((tl_container*)(vertices[i].aux_neighbors))->begin();
          uint64_t sum = 0;
          while (!it.done()) {
#if WEIGHTED
            auto v = (*it).first;
            auto w = (*it).second;
#else
            auto v = *it;
            sum += v;
            if (v != arr[edges_so_far]) {
              printf("\tIN BTREE LEVEL: vtx %lu, position %u, got ngh %u, should be %u\n", i, edges_so_far, v, arr[edges_so_far]);
            }
#endif
            ++edges_so_far;
            ++it;
				}
        ParallelTools::Reducer_sum<size_t> psum;
        // how to pass in psum rather than verify?
        ((tl_container*)(vertices[i].aux_neighbors))->parallel_map(i, [&psum](auto s, auto d) { psum += d; });
        ParallelTools::Reducer_sum<size_t> psum_early_exit;
        // how to pass in psum rather than verify?
        ((tl_container*)(vertices[i].aux_neighbors))->parallel_map_early_exit(i, [&psum_early_exit](auto s, auto d) { psum_early_exit += d; return false; });

        if (sum != psum.get()) {
          printf("vertex %lu, got psum %lu, should be %lu\n", i, psum.get(), sum);
        }
        if (sum != psum_early_exit.get()) {
           printf("vertex %lu, got psum early exit %lu, should be %lu\n", i, psum_early_exit.get(), sum);
        }        
      }
    }
  }

  template <class F>
  void inline TerraceGraph::map_neighbors_early_exit(size_t i, F &&f) const {
      uint32_t degree = vertices[i].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          if(f(i, v)) break;
          ++local_idx;  
        }
      } else { //degree > num_in_place
#if PREFETCH
        if (is_btree(i)) {
          __builtin_prefetch(vertices[i].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[i]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          if(f(i, v)) break;
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(i)) {
              __builtin_prefetch(((tl_container*)vertices[i].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[i].beginning]);
            }
          }
#endif
        }
        if (!is_btree(i)) {
          uint64_t idx = second_level.nodes[i].beginning + 1;
          uint64_t idx_end = second_level.nodes[i].end;
          while (idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if (v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
#endif

              // printf("map in PMA with early exit: (%lu, %lu)\n", i, v);
              if(f(i, v)) break;
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          auto it = ((tl_container*)(vertices[i].aux_neighbors))->begin();
          while (!it.done()) {
#if WEIGHTED
            auto v = (*it).first;
            auto w = (*it).second;
#else
            auto v = *it;
            if(f(i, v)) break;
#endif
            ++it;
				}
      }
    }
  }

  template <class F>
  void inline TerraceGraph::parallel_map_neighbors_early_exit(size_t i, F &&f) const {
      uint32_t degree = vertices[i].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          if(f(i, v)) break;
          ++local_idx;  
        }
      } else { //degree > num_in_place
#if PREFETCH
        if (is_btree(i)) {
          __builtin_prefetch(vertices[i].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[i]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[i].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[i].weights[local_idx];
#endif
          if(f(i, v)) break;
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(i)) {
              __builtin_prefetch(((tl_container*)vertices[i].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[i].beginning]);
            }
          }
#endif
        }
        if (!is_btree(i)) {
          uint64_t idx = second_level.nodes[i].beginning + 1;
          uint64_t idx_end = second_level.nodes[i].end;
          while (idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if (v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
#endif

              // printf("map in PMA with early exit: (%lu, %lu)\n", i, v);
              if(f(i, v)) break;
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          ((tl_container*)(vertices[i].aux_neighbors))->parallel_map_early_exit(i, f);
				}
      }
    }

  template <class F, typename VS>
    void inline TerraceGraph::map_sparse(F &f, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      //std::cout << "vertex: " << self_index << " degree: " << degree << '\n';
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v, w) == 1) {
#else
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
#endif
            if (output) {
              output_vs.insert_sparse(v);
            }
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (is_btree(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[self_index]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v, w) == 1) {
#else
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
#endif
            if (output) {
              output_vs.insert_sparse(v);
            }
          }
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(self_index)) {
              __builtin_prefetch(((tl_container*)vertices[self_index].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[self_index].beginning]);
            }
          }
#endif
        }
        if (!is_btree(self_index)) {
          uint64_t idx = second_level.nodes[self_index].beginning + 1;
          uint64_t idx_end = second_level.nodes[self_index].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
              if (f.cond(v) == 1 && f.updateAtomic(self_index, v, w) == 1) {
#else
              if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
#endif
                if (output) {
                  output_vs.insert_sparse(v);
                }
              }
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }

          }
        } else {
					tl_container* btree =
						(tl_container*)(vertices[self_index].aux_neighbors);
					struct Btree_map<F, VS> update_fn(output_vs, f, output, self_index, btree);
					btree->map(update_fn);
				}
      }
      /*
      auto it = neighbors(self_index);
      while (!it.done()) {
        // edge_t edge = *it;
        //// int32_t v = edge.dest;
        auto v  = *it;
        if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
          if (output) {
          vs.insert(v);
          }
        }
        ++it;
      }

      */
    }

  template <class F, typename VS>
    void inline TerraceGraph::map_dense_vs_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (f.update(v, self_index, w) == 1) {
#else
          if (f.update(v, self_index) == 1) {
#endif
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (is_btree(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[self_index]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (f.update(v, self_index, w) == 1) {
#else
          if (f.update(v, self_index) == 1) {
#endif
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx; 
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(self_index)) {
              __builtin_prefetch(((tl_container*)vertices[self_index].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[self_index].beginning]);
            }
          }
#endif
        }
        if (!is_btree(self_index)) {
          uint64_t idx = second_level.nodes[self_index].beginning + 1;
          uint64_t idx_end = second_level.nodes[self_index].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
              if (f.update(v, self_index, w) == 1) {
#else
              if (f.update(v, self_index) == 1) {
#endif
                if (output) {
                  output_vs.insert_dense(self_index);
                }
              }
              if (f.cond(self_index) == 0) {
                return;
              }
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }

          }
        } else {
          auto it = ((tl_container*)(vertices[self_index].aux_neighbors))->begin();
          while (!it.done()) {
#if WEIGHTED
            auto v = (*it).first;
            auto w = (*it).second;
            if (f.update(v, self_index, w) == 1) {
#else
            auto v = *it;
            if (f.update(v, self_index) == 1) {
#endif
              if (output) {
                output_vs.insert_dense(self_index);
              }
            }
            if (f.cond(self_index) == 0) {
              return;
            }
            ++it;
          }
        }
      }
    }

  template <class F, typename VS>
    void inline TerraceGraph::map_dense_vs_not_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (vs.has_dense_no_all(v) && f.update(v, self_index, w) == 1) {
#else
          if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
#endif
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (is_btree(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        } else {
          __builtin_prefetch(&second_level.nodes[self_index]);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
#if WEIGHTED
          auto w = vertices[self_index].weights[local_idx];
          if (vs.has_dense_no_all(v) && f.update(v, self_index, w) == 1) {
#else
          if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
#endif
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (is_btree(self_index)) {
              __builtin_prefetch(((tl_container*)vertices[self_index].aux_neighbors)->get_root());  
            } else {
              __builtin_prefetch(&second_level.edges.dests[second_level.nodes[self_index].beginning]);
            }
          }
#endif
        }
        if (!is_btree(self_index)) {
          uint64_t idx = second_level.nodes[self_index].beginning + 1;
          uint64_t idx_end = second_level.nodes[self_index].end;
          while ( idx < idx_end) {
            auto v = second_level.edges.dests[idx];
            if ( v != NULL_VAL) {
#if WEIGHTED
            auto w = second_level.edges.vals[idx];
              if (vs.has_dense_no_all(v) && f.update(v, self_index, w) == 1) {
#else
              if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
#endif
                if (output) {
                  output_vs.insert_dense(self_index);
                }
              }
              if (f.cond(self_index) == 0) {
                return;
              }
              idx++;
            } else {
              idx = ((idx >> second_level.edges.loglogN) +1 ) << (second_level.edges.loglogN);
            }
          }
        } else {
          auto it = ((tl_container*)(vertices[self_index].aux_neighbors))->begin();
          while (!it.done()) {
#if WEIGHTED
            auto v = (*it).first;
            auto w = (*it).second;
            if (vs.has_dense_no_all(v) && f.update(v, self_index, w) == 1) {
#else
            auto v = *it;
            if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
#endif
              if (output) {
                output_vs.insert_dense(self_index);
              }
            }
            if (f.cond(self_index) == 0) {
              return;
            }
            ++it;
          }
        }
      }
      /*
      auto it = neighbors(self_index);
      while (!it.done()) {
      #if WEIGHTED
        auto v = (*it).first;
        auto w = (*it).second;
        if (vs.has(v) && f.update(v, self_index, w) == 1) {
      #else
        auto v = *it;

        if (vs.has(v) && f.update(v, self_index) == 1) {
      #endif
            if (output) {
                vs.insert(self_index);
            }
        }
        if (f.cond(self_index) == 0) {
            return;
        }
        ++it;
      }
      */
    }
}
