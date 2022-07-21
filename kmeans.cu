/*
Copyright 2013  Bryan Catanzaro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <typeinfo>

// in previous kmeans.h
#include <thrust/reduce.h>

// in previous centroids.h
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

// in previous labels.h
#include <cfloat>
#include <cublas_v2.h>

#include "kmeans.h"

// set to 1 to print informations, set to 0 for performance
#define KMEANS_DEBUG 1
// 0.01%
#define EARLY_TERM_THRES 0.00001
#define MAX_KMEANS_ITER 50

#if KMEANS_DEBUG == 0
// given as parameters in gen_kclusters
static int kmeans_point_count;
static int kmeans_cluster_count;

#elif KMEANS_DEBUG == 1
#define PIC_WIDTH 2048
static int kmeans_point_count = 10000;
static int kmeans_cluster_count = 128;
#define DIM 2
#endif

#define ZERO_CHECK(x) (x < 0.000000000000001) ? 0 : x

// Compute the sum of a double array
// Call this kernel with blockDim=(1024, 1, 1)
__global__ void kmeans_double_reduction_kernel(volatile double *data, const int data_size) {
    // compact the data into the first blockDim.x positions
    int tid = threadIdx.x;

    // check for zeros
    if (tid < data_size) {
        if (data[tid] < 0.000000000000001)          // < 10^-15
            data[tid] = 0;
    }

    for (int idx = tid + blockDim.x; idx < data_size; idx += blockDim.x) {
        if (data[tid] > 0.000000000000001)
            data[tid] += data[idx];
    }

    if (data_size > 512 && tid + 512 < data_size) {
        data[tid] += data[tid + 512];
    }    
    if (data_size > 256 && tid + 256 < data_size) {
        data[tid] += data[tid + 256];
    }
    if (data_size > 128 && tid + 128 < data_size) {
        data[tid] += data[tid + 128];
    }    
    if (data_size > 64 && tid + 64 < data_size) {
        data[tid] += data[tid + 64];
    }
    if (data_size > 32 && tid + 32 < data_size) {
        data[tid] += data[tid + 32];
    }    
    if (data_size > 16 && tid + 16 < data_size) {
        data[tid] += data[tid + 16];
    }
    if (data_size > 8 && tid + 8 < data_size) {
        data[tid] += data[tid + 8];
    }    
    if (data_size > 4 && tid + 4 < data_size) {
        data[tid] += data[tid + 4];
    }
    if (data_size > 2 && tid + 2 < data_size) {
        data[tid] += data[tid + 2];
    }    
    if (data_size > 1 && tid + 1 < data_size) {
        data[tid] += data[tid + 1];
    }
}

//data: points in row-major order (kmeans_point_count rows, DIM cols)
//dots: result vector (kmeans_point_count rows). dots[tid] = x*x + y*y + ...
// NOTE:
//Memory accesses in this function are uncoalesced!!
//This is because data is in row major order
//However, in k-means, it's called outside the optimization loop
//on the large data array, and inside the optimization loop it's
//called only on a small array, so it doesn't really matter.
//If this becomes a performance limiter, transpose the data somewhere
template<typename T>
__global__ void self_dots(T* data, T* dots, int point_count) {
	T accumulator = 0;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_id < point_count) {
        for (int i = 0; i < DIM; i++) {
            T value = data[i + global_id * DIM];
            accumulator += value * value;
        }
        dots[global_id] = accumulator;
    }    
}

/**
 * @brief dots[tid] = x*x + y*y + ...
 * 
 * @param data 
 * @param dots 
 * @param data_size the point count / centroid count of data
 */
template<typename T>
void make_self_dots(thrust::device_vector<T>& data,
                    thrust::device_vector<T>& dots,
                    const int data_size) {
    self_dots<<<(data_size - 1)/256+1, 256>>>(thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(dots.data()), data_size);
}


/**
 * @brief dots(x, y) = data_dots[x] + centroid_dots[y], each block computes a 32x32 region
 * 
 * @param data_dots size = point_count
 * @param centroid_dots size = cluster_count
 * @param dots pairwise_distance, size = point_count * cluster_count
 */
template<typename T>
__global__ void all_dots(T* data_dots, T* centroid_dots, T* dots, int point_count, int cluster_count) {
	__shared__ T local_data_dots[32];
	__shared__ T local_centroid_dots[32];

    // copy 32 elements into local array
    int data_index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((data_index < point_count) && (threadIdx.y == 0)) {
        local_data_dots[threadIdx.x] = data_dots[data_index];
    }
    
    // copy 32 elements into local array, use thread 1 to parallelize the copy with thread 0 (above)
    int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
    if ((centroid_index < cluster_count) && (threadIdx.y == 1)) {
        local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
    }

   	__syncthreads();

    // in a global aspect, dots(x, y) = data_dots[x] + centroid_dots[y]
	centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
    if ((data_index < point_count) && (centroid_index < cluster_count)) {
        dots[data_index + centroid_index * point_count] = local_data_dots[threadIdx.x] +
            local_centroid_dots[threadIdx.y];
    }
}

/**
 * @brief dots(x, y) = data_dots[x] + centroid_dots[y], each block computes a 32x32 region
 * 
 * @param data_dots size = kmeans_point_count
 * @param centroid_dots size = kmeans_cluster_count
 * @param dots pairwise_distance, size = kmeans_point_count * kmeans_cluster_count
 */
template<typename T>
void make_all_dots(thrust::device_vector<T>& data_dots,
                   thrust::device_vector<T>& centroid_dots,
                   thrust::device_vector<T>& dots) {
    dim3 gridDim = dim3((kmeans_point_count - 1)/32 + 1, (kmeans_cluster_count - 1)/32 + 1);
    dim3 blockDim = dim3(32, 32);
    all_dots<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(data_dots.data()),
        thrust::raw_pointer_cast(centroid_dots.data()),
        thrust::raw_pointer_cast(dots.data()), kmeans_point_count, kmeans_cluster_count);
}

struct cublas_state {
    cublasHandle_t cublas_handle;
    cublas_state() {
        cublasStatus_t stat;
        stat = cublasCreate(&cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS initialization failed" << std::endl;
            exit(1);
        }
    }
    ~cublas_state() {
        cublasStatus_t stat;
        stat = cublasDestroy(cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS destruction failed" << std::endl;
            exit(1);
        }
    }
};


cublas_state state;

// single precision gemm
void gemm(cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k, const float *alpha,
          const float *A, int lda, const float *B, int ldb,
          const float *beta, float *C, int ldc) {
    cublasStatus_t status = cublasSgemm(state.cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda, B, ldb,
                                        beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Sgemm" << std::endl;
        exit(1);
    }
}

// double precision gemm
void gemm(cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k, const double *alpha,
          const double *A, int lda, const double *B, int ldb,
          const double *beta, double *C, int ldc) {
    cublasStatus_t status = cublasDgemm(state.cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda, B, ldb,
                                        beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Dgemm" << std::endl;
        exit(1);
    }
}


/**
 * @brief results are stored in pairwise_distance
 * 
 * @param data size = (kmeans_point_count * DIM)
 * @param centroids size = (kmeans_cluster_count * DIM)
 * @param data_dots size = (kmeans_point_count)
 * @param centroid_dots size = (kmeans_cluster_count)
 * @param pairwise_distances size = (kmeans_point_count * kmeans_cluster_count)
 */
template<typename T>
void calculate_distances(thrust::device_vector<T>& data,
                         thrust::device_vector<T>& centroids,
                         thrust::device_vector<T>& data_dots,
                         thrust::device_vector<T>& centroid_dots,
                         thrust::device_vector<T>& pairwise_distances) {
    make_self_dots(centroids, centroid_dots, kmeans_cluster_count);
    make_all_dots(data_dots, centroid_dots, pairwise_distances);
    // C = alpha * A * B + beta * C
    //||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
    //pairwise_distances initially has ||x||^2 + ||y||^2, so beta = 1.0
    //The gemm calculates x.y for all x and y, so alpha = -2.0
    T alpha = -2.0;
    T beta = 1.0;
    //If the data were in standard column major order, we'd do a
    //centroids * data ^ T
    //But the data is in row major order, so we have to permute
    //the arguments a little
    gemm(CUBLAS_OP_T, CUBLAS_OP_N,
         kmeans_point_count, kmeans_cluster_count, DIM, &alpha,
         thrust::raw_pointer_cast(data.data()),
         DIM,//Has to be n or d
         thrust::raw_pointer_cast(centroids.data()),
         DIM,//Has to be k or d
         &beta,
         thrust::raw_pointer_cast(pairwise_distances.data()),
         kmeans_point_count); //Has to be n or k
}

// Each thread traverse all centroid and determine the nearest centroid to one point.
// The distance to the new centroid is stored in array distances.
// Atomic add the count of changes to the global variable changes, which should be comment out for performance.
template<typename T>
__global__ void make_new_labels(T* pairwise_distances, int* labels, int* changes, T* distances,\
    int point_count, int cluster_count) {
    T min_distance = DBL_MAX;
    T min_idx = -1;
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < point_count) {
        int old_label = labels[global_id];
        for(int c = 0; c < cluster_count; c++) {
            T distance = ZERO_CHECK(pairwise_distances[c * point_count + global_id]);
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = c;
            }
        }
        labels[global_id] = min_idx;

        distances[global_id] = sqrt(ZERO_CHECK(min_distance));
        if (old_label != min_idx) {
            atomicAdd(changes, 1);
        }
    }
}

// Call a kernel function to recalculate the nearest centroid to each point.
// Return the count of points that changed their label (nearest centroid).
template<typename T>
int relabel(thrust::device_vector<T>& pairwise_distances,
            thrust::device_vector<int>& labels,
            thrust::device_vector<T>& distances) {
    thrust::device_vector<int> changes(1);
    changes[0] = 0;
    dim3 gridDim = dim3((kmeans_point_count - 1) / 256 + 1);
    dim3 blockDim = dim3(256);
    make_new_labels<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(pairwise_distances.data()),
        thrust::raw_pointer_cast(labels.data()),
        thrust::raw_pointer_cast(changes.data()),
        thrust::raw_pointer_cast(distances.data()), kmeans_point_count, kmeans_cluster_count);
    return changes[0];
}

/**
 * @brief Add the value in accumulator to target (a coord of centroid)
 * 
 * @param label the index of the centroid (add the dimension to label, to get the index of a coord)
 * @param dimension current coord's dimension
 * @param accumulator the value to be added to the target (a coord of the centroid)
 * @param centroids the coord array of centroids
 * @param count record how many points are classified to this centroid
 * @param counts_cen the count array of each centroid
 */
template<typename T>
__device__ __forceinline__
void update_centroid(int label, int dimension,
                     T accumulator, T* centroids,
                     int count, int* counts_cen) {
    int index = label * DIM + dimension;
    T* target = centroids + index;
    atomicAdd(target, accumulator);
    if (dimension == 0) {
        atomicAdd(counts_cen + label, count);
    }             
}

/**
 * @brief This function only adds up the point coords but not divide the count.
 * threadIdx.x is used to represent the dimension of the same point
 * threadIdx.y is used to represent the point index
 * 
 * @param data point coord
 * @param ordered_labels labels of each point
 * @param ordered_indices data array did not change the order, use this idx array to access data array
 * @param centroids output, the new centroids
 * @param counts_cen output, the point count in each cluster
 */
template<typename T>
__global__ void calculate_centroids(T* data, int* ordered_labels,
    int* ordered_indices, T* centroids, int* counts_cen, int point_count) {
    int in_flight = blockDim.y * gridDim.y;
    int labels_per_row = (point_count - 1) / in_flight + 1;

    // accumulate the coord in each dimension
    for(int dimension = threadIdx.x; dimension < DIM; dimension += blockDim.x) {
        T accumulator = 0;
        int count = 0;
        int global_id = threadIdx.y + blockIdx.y * blockDim.y;      // index of point
        int start = global_id * labels_per_row;
        int end = (global_id + 1) * labels_per_row;
        end = (end > point_count) ? point_count : end;
        int prior_label;
        if (start < point_count) {
            prior_label = ordered_labels[start];
        
            for(int label_number = start; label_number < end; label_number++) {
                int label = ordered_labels[label_number];
                // a few points in prior have defferent label as the current point, 
                // process the accumulator and the count, then reset it
                if (label != prior_label) {
                    update_centroid(prior_label, dimension,
                                    accumulator, centroids, count, counts_cen);
                    accumulator = 0;
                    count = 0;
                }
  
                T value = data[dimension + ordered_indices[label_number] * DIM];
                accumulator += value;
                prior_label = label;
                count++;
            }
            update_centroid(prior_label, dimension,
                            accumulator, centroids, count, counts_cen);
        }
    }
}

// each coordinate of a centroid is divided by its point count
// if a cluster has 3 points, the coord will become (x/3, y/3)
template<typename T>
__global__ void scale_centroids(int* counts, T* centroids, int cluster_count) {
    int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((global_id_x < DIM) && (global_id_y < cluster_count)) {
        int count = counts[global_id_y];
        //To avoid introducing divide by zero errors
        //If a centroid has no weight, we'll do no normalization
        //This will keep its coordinates defined.
        if (count < 1) {
            count = 1;
        }
        double scale = 1.0/double(count);
        centroids[global_id_x + DIM * global_id_y] *= scale;
    }
}

/**
 * @brief 
 * 
 * @param data 
 * @param labels 
 * @param centroids Stores the result coordinates
 * @param counts Stores the point count in each cluster
 */
template<typename T>
void find_centroids(thrust::device_vector<T>& data,
                    //Labels are taken by value because
                    //they get destroyed in sort_by_key
                    //So we need to make a copy of them
                    thrust::device_vector<int> labels,
                    thrust::device_vector<T>& centroids,
                    thrust::device_vector<int>& point_counts) {
    thrust::device_vector<int> indices(kmeans_point_count);
    // reset point_counts to all zeros
    thrust::fill(point_counts.begin(), point_counts.end(), 0);

    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(kmeans_point_count),
                 indices.begin());
    //Bring all labels with the same value together
    thrust::sort_by_key(labels.begin(),
                        labels.end(),
                        indices.begin());

    //Initialize centroids to all zeros
    thrust::fill(centroids.begin(),
                 centroids.end(),
                 0);
    
    //Calculate centroids 
    int n_threads_x = 64;
    int n_threads_y = 16;
    //XXX Number of blocks here is hard coded at 30
    //This should be taken care of more thoughtfully.
    calculate_centroids<<<dim3(1, 30), dim3(n_threads_x, n_threads_y)>>>
        (thrust::raw_pointer_cast(data.data()),
         thrust::raw_pointer_cast(labels.data()),
         thrust::raw_pointer_cast(indices.data()),
         thrust::raw_pointer_cast(centroids.data()),
         thrust::raw_pointer_cast(point_counts.data()), kmeans_point_count);
    
    //Scale centroids, because the calculate_centroid kernel just add the coord up.
    scale_centroids<<<dim3((DIM - 1)/32 + 1, (kmeans_cluster_count - 1)/32 + 1), dim3(32, 32)>>>
        (thrust::raw_pointer_cast(point_counts.data()),
         thrust::raw_pointer_cast(centroids.data()), kmeans_cluster_count);
}



//! kmeans clusters data into k groups
/*! 
  
  \param iterations How many iterations to run
  \param data Data points, in row-major order. This vector must have
  size n * d, and since it's in row-major order, data point x occupies
  positions [x * d, (x + 1) * d) in the vector. The vector is passed
  by reference since it is shared with the caller and not copied.
  \param labels Cluster labels. This vector has size n.
  The vector is passed by reference since it is shared with the caller
  and not copied.
  \param centroids Centroid locations, in row-major order. This
  vector must have size k * d, and since it's in row-major order,
  centroid x occupies positions [x * d, (x + 1) * d) in the
  vector. The vector is passed by reference since it is shared
  with the caller and not copied.
  \param distances Distances from points to centroids. This vector has
  size n. It is passed by reference since it is shared with the caller
  and not copied.
  \param init_from_labels If true, the labels need to be initialized
  before calling kmeans. If false, the centroids need to be
  initialized before calling kmeans. Defaults to true, which means
  the labels must be initialized.
  \param threshold This controls early termination of the kmeans
  iterations. When the sum of distances from points to centroids 
  between two iterations changes by less than the threshold, 
  the iterations are terminated.
  \return The number of iterations actually performed.
*/
template<typename T>
int kmeans(int iterations,
           thrust::device_vector<T>& data,
           thrust::device_vector<int>& labels,
           thrust::device_vector<T>& centroids,
           thrust::device_vector<T>& distances,
           bool init_from_labels=true,
           double threshold=EARLY_TERM_THRES) {
    thrust::device_vector<T> data_dots(kmeans_point_count);
    // in the original version, the size of centroid_dots is kmeans_point_count (n)
    thrust::device_vector<T> centroid_dots(kmeans_cluster_count);
    thrust::device_vector<int> point_counts(kmeans_cluster_count);      // the number of points in each cluster
    thrust::device_vector<T> pairwise_distances(kmeans_point_count * kmeans_cluster_count);
    
    make_self_dots(data, data_dots, kmeans_point_count);

    if (init_from_labels) {
        find_centroids(data, labels, centroids, point_counts);
    }   
    T prior_distance_sum = 0;
    int i = 0;
    for(; i < iterations; i++) {
        calculate_distances(data, centroids, data_dots,
            centroid_dots, pairwise_distances);
        int changes = relabel(pairwise_distances, labels, distances);
        find_centroids(data, labels, centroids, point_counts);
        T distance_sum = thrust::reduce(distances.begin(), distances.end());

        // kmeans_double_reduction_kernel<<<dim3(1, 1, 1), dim3(1024, 1, 1)>>>
        //     (thrust::raw_pointer_cast(distances.data()), kmeans_point_count);

#if KMEANS_DEBUG == 1
        std::cout << "Iteration " << i << " produced " << changes
                  << " changes, and total distance is " << distance_sum << std::endl;
#endif

        // early terminating condition
        if (i > 0) {
            T delta;
            if (prior_distance_sum == 0)
                delta = 0;
            else
                delta = abs((distance_sum - prior_distance_sum) / prior_distance_sum);

            if (delta < threshold) {
#if KMEANS_DEBUG == 1
                std::cout << "Threshold triggered, terminating iterations early" << std::endl;
                // debug, print the point count in each cluster
                thrust::host_vector<int> point_counts_h = point_counts;      // the number of points in each cluster
                int *point_counts_raw = thrust::raw_pointer_cast(point_counts_h.data());
#endif
                return i + 1;
            }
        }
        prior_distance_sum = distance_sum;
    }

#if KMEANS_DEBUG == 1
    // debug, print the point count in each cluster
    for (int cl = 0; cl < kmeans_cluster_count; ++cl) {
        printf("Number of points in cluster %d: %d\n", cl, (int)point_counts[cl]);
    }
#endif

    return i;
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
    thrust::host_vector<T> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (T)rand() / (T)RAND_MAX * (T)PIC_WIDTH;
    }
    array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}

// Use kmeans++ method to init the centroids
template<typename T>
void kmeans_plus_plus(T *data, T *centroids) {
    // probility to be a centroid of each point
    double prob[kmeans_point_count];
    // stores the distance of each point to the nearest centroid
    double dist[kmeans_point_count];
    double total_dist;
    // init prob of all point to 1/kmeans_point_count
    for (int i = 0; i < kmeans_point_count; ++i) {
        prob[i] = (double) 1 / (double) kmeans_point_count;
    }

    // randomly choose the first centroid
    double ran_num = (T)rand() / (T)RAND_MAX;
    for (int i = 0; i < kmeans_point_count; ++i) {
        ran_num -= prob[i];
        if (ran_num <= 0) {     // select the point that make the ran_num less than 0
            for (int d = 0; d < DIM; ++d)
                centroids[0 * DIM + d] = data[i * DIM + d];
            break;
        }
    }

    // compute the dist from each point to the first centroid
    total_dist = 0;
    for (int i = 0; i < kmeans_point_count; ++i) {
        dist[i] = 0;
        for (int d = 0; d < DIM; ++d)
            dist[i] += (data[i * DIM + d] - centroids[0 * DIM + d]) * (data[i * DIM + d] - centroids[0 * DIM + d]);
        total_dist += dist[i];
    }

    // compute the remaining clusters
    for (int cl = 1; cl < kmeans_cluster_count; ++cl) {

        for (int d = 0; d < DIM; ++d)       // sentinel value
            centroids[cl * DIM + d] = -1;

        // change the prob based on the dist
        for (int i = 0; i < kmeans_point_count; ++i) {
            prob[i] = dist[i] / total_dist;
        }

        ran_num = (T)rand() / (T)RAND_MAX;      // a new random number
        for (int i = 0; i < kmeans_point_count; ++i) {
            ran_num -= prob[i];
            if (ran_num <= 0) {     // select the point that make the ran_num less than 0
                for (int d = 0; d < DIM; ++d)
                    centroids[cl * DIM + d] = data[i * DIM + d];
                break;
            }
        }

        if (centroids[cl * DIM + 0] == -1) {    // if did init the centroid successfully, randomly pick coords
            for (int d = 0; d < DIM; ++d)
                centroids[cl * DIM + d] = (T)rand() / (T)RAND_MAX * (T)PIC_WIDTH;
        }

        // compute the dist from each point to the nearest centroid
        total_dist = 0;
        for (int i = 0; i < kmeans_point_count; ++i) {
            dist[i] = DBL_MAX;
            for (int cur_cl = 0; cur_cl <= cl; ++cur_cl) {
                double cur_dist = 0;
                for (int d = 0; d < DIM; ++d)
                    cur_dist += (data[i * DIM + d] - centroids[cur_cl * DIM + d]) * (data[i * DIM + d] - centroids[cur_cl * DIM + d]);
                if (cur_dist < dist[i]) {
                    dist[i] = cur_dist;
                }
            }
            total_dist += dist[i];
        }
    }
}

/**
 * @brief generate k clusters using kmeans method, the data_raw is float or double
 * The centroids of each clusters are initialized with kmeans++ method
 * 
 * @param data_raw input data, its size is [kmeans_point_count * DIM]
 * @param labels_raw output data, assign a label to each data point
 */
template<typename T>
void gen_kclusters(T *data_raw, int* labels_raw, int kmeans_point_count_in, int kmeans_cluster_count_in) {
    if (kmeans_point_count_in == 0) {
        printf("Point count is 0, return directly\n");
        return;
    }
    kmeans_point_count = kmeans_point_count_in;
    kmeans_cluster_count = kmeans_cluster_count_in;
	T centroids_raw[kmeans_cluster_count * DIM];
    // Use kmeans++ to init centroids_raw
    kmeans_plus_plus(data_raw, centroids_raw);
	
	thrust::device_vector<double> data(data_raw, data_raw + kmeans_point_count * DIM);
    thrust::device_vector<double> centroids(centroids_raw, centroids_raw + kmeans_cluster_count * DIM);
    thrust::device_vector<int> labels(kmeans_point_count);
    thrust::device_vector<double> distances(kmeans_point_count);

    kmeans(MAX_KMEANS_ITER, data, labels, centroids, distances, false);

    for(int i = 0; i < kmeans_point_count; ++i)
        labels_raw[i] = labels[i];
}


#if KMEANS_DEBUG == 1
template<typename T>
void test() {
    int iterations = MAX_KMEANS_ITER;

    if (kmeans_point_count == 0) {
        printf("Point count is 0, return directly\n");
    }

    T data_raw[kmeans_point_count * DIM];
    // init data_raw with random number
    for(int i = 0; i < kmeans_point_count * DIM; i++) {
        data_raw[i] = (T)rand() / (T)RAND_MAX * (T)PIC_WIDTH;
    }

	T centroids_raw[kmeans_cluster_count * DIM];
    // Use kmeans++ to init centroids_raw
    kmeans_plus_plus(data_raw, centroids_raw);
	
	thrust::device_vector<double> data(data_raw, data_raw + kmeans_point_count * DIM);
    thrust::device_vector<double> centroids(centroids_raw, centroids_raw + kmeans_cluster_count * DIM);
    thrust::device_vector<int> labels(kmeans_point_count);
    thrust::device_vector<double> distances(kmeans_point_count);
    
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Precision: " << typeid(T).name() << std::endl;

    int i = kmeans(iterations, data, labels, centroids, distances, false);

    std::cout << "Performed " << i << " iterations" << std::endl;
}

int main() {
    test<double>();
    // test<float>();
}
#endif