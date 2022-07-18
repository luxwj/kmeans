
#ifndef KMEANS_H
#define KMEANS_H

/**
 * @brief generate k clusters using kmeans method
 * The centroids of each clusters are initialized with kmeans++ method
 * 
 * @param data_raw input data, its size is [SITES_NUMBER * DIM]
 * @param labels_raw output data, assign a label to each data point
 */
template<typename T>
void gen_kclusters(T *data_raw, int* labels_raw, int kmeans_point_count, int kmeans_cluster_count);

#endif