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

#define SITES_NUMBER 5
#define DIM 3
#define KMEANS_K 2

//data: points in row-major order (SITES_NUMBER rows, DIM cols)
//dots: result vector (SITES_NUMBER rows). dots[tid] = x*x + y*y + ...
// NOTE:
//Memory accesses in this function are uncoalesced!!
//This is because data is in row major order
//However, in k-means, it's called outside the optimization loop
//on the large data array, and inside the optimization loop it's
//called only on a small array, so it doesn't really matter.
//If this becomes a performance limiter, transpose the data somewhere
template<typename T>
__global__ void self_dots(T* data, T* dots) {
	T accumulator = 0;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_id < SITES_NUMBER) {
        for (int i = 0; i < DIM; i++) {
            T value = data[i + global_id * DIM];
            accumulator += value * value;
        }
        dots[global_id] = accumulator;
    }    
}

// dots[tid] = x*x + y*y + ...
template<typename T>
void make_self_dots(thrust::device_vector<T>& data,
                    thrust::device_vector<T>& dots) {
    self_dots<<<(SITES_NUMBER - 1)/256+1, 256>>>(thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(dots.data()));
}


/**
 * @brief dots(x, y) = data_dots[x] + centroid_dots[y], each block computes a 32x32 region
 * 
 * @param data_dots size = (SITES_NUMBER)
 * @param centroid_dots size = (KMEANS_K)
 * @param dots pairwise_distance, size = (SITES_NUMBER * KMEANS_K)
 */
template<typename T>
__global__ void all_dots(T* data_dots, T* centroid_dots, T* dots) {
	__shared__ T local_data_dots[32];
	__shared__ T local_centroid_dots[32];

    // copy 32 elements into local array
    int data_index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((data_index < SITES_NUMBER) && (threadIdx.y == 0)) {
        local_data_dots[threadIdx.x] = data_dots[data_index];
    }
    
    // copy 32 elements into local array, use thread 1 to parallelize the copy with thread 0 (above)
    int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
    if ((centroid_index < KMEANS_K) && (threadIdx.y == 1)) {
        local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
    }

   	__syncthreads();

    // in a global aspect, dots(x, y) = data_dots[x] + centroid_dots[y]
	centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
    if ((data_index < SITES_NUMBER) && (centroid_index < KMEANS_K)) {
        dots[data_index + centroid_index * SITES_NUMBER] = local_data_dots[threadIdx.x] +
            local_centroid_dots[threadIdx.y];
    }
}

/**
 * @brief dots(x, y) = data_dots[x] + centroid_dots[y], each block computes a 32x32 region
 * 
 * @param data_dots size = (SITES_NUMBER)
 * @param centroid_dots size = (KMEANS_K)
 * @param dots pairwise_distance, size = (SITES_NUMBER * KMEANS_K)
 */
template<typename T>
void make_all_dots(thrust::device_vector<T>& data_dots,
                   thrust::device_vector<T>& centroid_dots,
                   thrust::device_vector<T>& dots) {
    dim3 gridDim = dim3((SITES_NUMBER - 1)/32 + 1, (KMEANS_K - 1)/32 + 1);
    dim3 blockDim = dim3(32, 32);
    all_dots<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(data_dots.data()),
        thrust::raw_pointer_cast(centroid_dots.data()),
        thrust::raw_pointer_cast(dots.data()));
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
 * @param data size = (SITES_NUMBER * DIM)
 * @param centroids size = (KMEANS_K * DIM)
 * @param data_dots size = (SITES_NUMBER)
 * @param centroid_dots size = (KMEANS_K)
 * @param pairwise_distances size = (SITES_NUMBER * KMEANS_K)
 */
template<typename T>
void calculate_distances(thrust::device_vector<T>& data,
                         thrust::device_vector<T>& centroids,
                         thrust::device_vector<T>& data_dots,
                         thrust::device_vector<T>& centroid_dots,
                         thrust::device_vector<T>& pairwise_distances) {
    make_self_dots(centroids, centroid_dots);
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
         SITES_NUMBER, KMEANS_K, DIM, &alpha,
         thrust::raw_pointer_cast(data.data()),
         DIM,//Has to be n or d
         thrust::raw_pointer_cast(centroids.data()),
         DIM,//Has to be k or d
         &beta,
         thrust::raw_pointer_cast(pairwise_distances.data()),
         SITES_NUMBER); //Has to be n or k
}

template<typename T>
__global__ void make_new_labels(int n, int k, T* pairwise_distances,
                                int* labels, int* changes,
                                T* distances) {
    T min_distance = DBL_MAX;
    T min_idx = -1;
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < n) {
        int old_label = labels[global_id];
        for(int c = 0; c < k; c++) {
            T distance = pairwise_distances[c * n + global_id];
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = c;
            }
        }
        labels[global_id] = min_idx;
        distances[global_id] = sqrt(min_distance);
        if (old_label != min_idx) {
            atomicAdd(changes, 1);
        }
    }
}


template<typename T>
int relabel(int n, int k,
            thrust::device_vector<T>& pairwise_distances,
            thrust::device_vector<int>& labels,
            thrust::device_vector<T>& distances) {
    thrust::device_vector<int> changes(1);
    changes[0] = 0;
    make_new_labels<<<(n-1)/256+1,256>>>(
        n, k,
        thrust::raw_pointer_cast(pairwise_distances.data()),
        thrust::raw_pointer_cast(labels.data()),
        thrust::raw_pointer_cast(changes.data()),
        thrust::raw_pointer_cast(distances.data()));
    return changes[0];
}

template<typename T>
__device__ __forceinline__
void update_centroid(int label, int dimension,
                     int d,
                     T accumulator, T* centroids,
                     int count, int* counts) {
    int index = label * d + dimension;
    T* target = centroids + index;
    atomicAdd(target, accumulator);
    if (dimension == 0) {
        atomicAdd(counts + label, count);
    }             
}

// threadIdx.x is used to represent the dimension of the same point
// threadIdx.y is used to represent the point index
template<typename T>
__global__ void calculate_centroids(T* data, int* ordered_labels,
    int* ordered_indices, T* centroids, int* counts) {
    int in_flight = blockDim.y * gridDim.y;
    int labels_per_row = (SITES_NUMBER - 1) / in_flight + 1; 
    for(int dimension = threadIdx.x; dimension < DIM; dimension += blockDim.x) {
        T accumulator = 0;
        int count = 0;
        int global_id = threadIdx.y + blockIdx.y * blockDim.y;
        int start = global_id * labels_per_row;
        int end = (global_id + 1) * labels_per_row;
        end = (end > SITES_NUMBER) ? SITES_NUMBER : end;
        int prior_label;
        if (start < SITES_NUMBER) {
            prior_label = ordered_labels[start];
        
            for(int label_number = start; label_number < end; label_number++) {
                int label = ordered_labels[label_number];
                if (label != prior_label) {
                    update_centroid(prior_label, dimension, DIM,
                                    accumulator, centroids, count, counts);
                    accumulator = 0;
                    count = 0;
                }
  
                T value = data[dimension + ordered_indices[label_number] * DIM];
                accumulator += value;
                prior_label = label;
                count++;
            }
            update_centroid(prior_label, dimension, DIM,
                            accumulator, centroids, count, counts);
        }
    }
}

// each coordinate of a centroid is divided by its point count
// if a cluster has 3 points, the coord will become (x/3, y/3)
template<typename T>
__global__ void scale_centroids(int* counts, T* centroids) {
    int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((global_id_x < DIM) && (global_id_y < KMEANS_K)) {
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

// the result coordinates are stored in centroids, 
template<typename T>
void find_centroids(thrust::device_vector<T>& data,
                    //Labels are taken by value because
                    //they get destroyed in sort_by_key
                    //So we need to make a copy of them
                    thrust::device_vector<int> labels,
                    thrust::device_vector<T>& centroids) {
    thrust::device_vector<int> indices(SITES_NUMBER);
    thrust::device_vector<int> counts(KMEANS_K);

    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(SITES_NUMBER),
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
         thrust::raw_pointer_cast(counts.data()));
    
    //Scale centroids
    scale_centroids<<<dim3((DIM - 1)/32 + 1, (KMEANS_K - 1)/32 + 1), dim3(32, 32)>>>
        (thrust::raw_pointer_cast(counts.data()),
         thrust::raw_pointer_cast(centroids.data()));
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
  iterations. If the ratio of the sum of distances from points to
  centroids from this iteration to the previous iteration changes by
  less than the threshold, than the iterations are
  terminated. Defaults to 0.000001
  \return The number of iterations actually performed.
*/
template<typename T>
int kmeans(int iterations,
           thrust::device_vector<T>& data,
           thrust::device_vector<int>& labels,
           thrust::device_vector<T>& centroids,
           thrust::device_vector<T>& distances,
           bool init_from_labels=true,
           double threshold=0.000001) {
    thrust::device_vector<T> data_dots(SITES_NUMBER);
    // in the original version, the size of centroid_dots is SITES_NUMBER (n)
    thrust::device_vector<T> centroid_dots(KMEANS_K);
    thrust::device_vector<T> pairwise_distances(SITES_NUMBER * KMEANS_K);
    
    make_self_dots(data, data_dots);

    if (init_from_labels) {
        find_centroids(data, labels, centroids);
    }   
    T prior_distance_sum = 0;
    int i = 0;
    for(; i < iterations; i++) {
        calculate_distances(data, centroids, data_dots,
            centroid_dots, pairwise_distances);

        int changes = relabel(SITES_NUMBER, KMEANS_K, pairwise_distances, labels, distances);
       
        
        find_centroids(data, labels, centroids);
        T distance_sum = thrust::reduce(distances.begin(), distances.end());
        std::cout << "Iteration " << i << " produced " << changes
                  << " changes, and total distance is " << distance_sum << std::endl;

        if (i > 0) {
            T delta = distance_sum / prior_distance_sum;
            if (delta > 1 - threshold) {
                std::cout << "Threshold triggered, terminating iterations early" << std::endl;
                return i + 1;
            }
        }
        prior_distance_sum = distance_sum;
    }
    return i;
}

template<typename T>
void print_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            typename T::value_type value = array[i * n + j];
            std::cout << value << ' ';
            // printf("%f ", value);
        }
        std::cout << std::endl;
    }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
    thrust::host_vector<T> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (T)rand()/(T)RAND_MAX;
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


void tiny_test() {
    int iterations = 3;

    thrust::device_vector<double> data(SITES_NUMBER * DIM);
    thrust::device_vector<int> labels(SITES_NUMBER);
    thrust::device_vector<double> centroids(KMEANS_K * DIM);
    thrust::device_vector<double> distances(SITES_NUMBER);
    
    // debug
    printf("Before generating data\n");

    // generate data (point coordinates)
    for(int i = 0; i < SITES_NUMBER; i++) {
        for(int d = 0; d < DIM; d++) {
            data[i * DIM + d] = (i % 2) * 3 + d;
        }
    }

    // debug
    printf("After generating data\n");

    std::cout << "Data: " << std::endl;
    print_array(data, SITES_NUMBER, DIM);

    labels[0] = 0;
    labels[1] = 0;
    labels[2] = 0;
    labels[3] = 1;
    labels[4] = 1;

    std::cout << "Labels: " << std::endl;
    print_array(labels, SITES_NUMBER, 1);
    
    // debug
    printf("before kmeans\n");

    int i = kmeans(iterations, data, labels, centroids, distances);
    std::cout << "Performed " << i << " iterations" << std::endl;

    std::cout << "Labels: " << std::endl;
    print_array(labels, SITES_NUMBER, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, KMEANS_K, DIM);

    std::cout << "Distances:" << std::endl;
    print_array(distances, SITES_NUMBER, 1);

}


void more_tiny_test() {
	double dataset[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5,
		1.1, 1.2,
		0.5, 15.5,
		1.5, 15.5,
		1.5, 16.5,
		0.5, 16.5,
		1.2, 16.1,
		15.5, 15.5,
		16.5, 15.5,
		16.5, 16.5,
		15.5, 16.5,
		15.6, 16.2,
		15.5, 0.5,
		16.5, 0.5,
		16.5, 1.5,
		15.5, 1.5,
		15.7, 1.6};
	double centers[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5};
	 
    int iterations = 3;
    int n = 20;
    int d = 2;
    int k = 4;
	
	thrust::device_vector<double> data(dataset, dataset+n*d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(centers, centers+k*d);
    thrust::device_vector<double> distances(n);
    
    kmeans(iterations, data, labels, centroids, distances, false);

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, k, d);

}

template<typename T>
void huge_test() {

    int iterations = 50;
    int n = 1e5;
    int d = 64;
    int k = 128;

    thrust::device_vector<T> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<T> centroids(k * d);
    thrust::device_vector<T> distances(n);
    
    std::cout << "Generating random data" << std::endl;
    std::cout << "Number of points: " << n << std::endl;
    std::cout << "Number of dimensions: " << d << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Precision: " << typeid(T).name() << std::endl;
    
    random_data(data, n, d);
    random_labels(labels, n, k);
    kmeans(iterations, data, labels, centroids, distances);
}

int main() {
    std::cout << "Input a character to choose a test:" << std::endl;
    std::cout << "Tiny test: t" << std::endl;
    std::cout << "More tiny test: m" << std::endl;
    std::cout << "Huge test: h: " << std::endl;
    char c = 't';

    switch (c) {
    case 't':
        tiny_test();
        exit(0);
    case 'm':
        more_tiny_test();
        exit(0);
    case 'h':
        break;
    default:
        std::cout << "Choice not understood, running huge test" << std::endl;
    }
    std::cout << "Double precision (d) or single precision (f): " << std::endl;
    std::cin >> c;
    switch(c) {
    case 'd':
        huge_test<double>();
        exit(0);
    case 'f':
        break;
    default:
        std::cout << "Choice not understood, running single precision"
                  << std::endl;
    }
    huge_test<float>();
    
}
