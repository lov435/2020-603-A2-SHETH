#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <list>
#include <map>
#include "../libarff/arff_parser.h"
#include "../libarff/arff_data.h"

using namespace std;

struct Neighbor {
	float distance;
	int cls;
};

struct Instance {
	float * attribs;
	int cls;
};

int majorityVote(list<Neighbor> & neighbors) {
    std::map<int, int> frequencyMap;
    int maxFrequency = 0;
    int  mostFrequentClass = -1;
	//cout << "Neighbors are " << endl;
    for (Neighbor neighbor : neighbors)
    {
		//cout << neighbor.distance << "  " << neighbor.cls << endl;
        int f = ++frequencyMap[neighbor.cls];
        if (f > maxFrequency)
        {
            maxFrequency = f;
            mostFrequentClass = neighbor.cls;
        }
    }

    return mostFrequentClass;
}

void predictFromDistances (float * distances, Instance * instances, int numInstances,
			int k, int * predictions) {

	//Initialize an empty neighbor
	Neighbor  neighbor;
	neighbor.distance = FLT_MAX;
	neighbor.cls = -1;
	//List of k neighbors
	std::list<Neighbor> neighbors (k, neighbor);

	for(int i=0; i < numInstances * numInstances; i++) {
		if(i%numInstances == i/numInstances)
			distances[i] = FLT_MAX; //subject and target instances are the same.

		for (std::list<Neighbor>::iterator it = neighbors.begin(); it != neighbors.end(); it++) {
			if(distances[i] < (*it).distance) {
				Neighbor neighbor;
				neighbor.distance = distances[i];
				neighbor.cls = instances[i%numInstances].cls;
				neighbors.insert(it, neighbor);
				neighbors.pop_back(); //Remove the last neighbor
				break;
			}
		}

		if((i+1)%numInstances == 0) {
			predictions[i/numInstances] = majorityVote(neighbors);
			//Reset the neighbors as we are starting the next instance
			neighbors.clear();
			for (int x=0; x <k; x++)
			{
				Neighbor neighbor;
				neighbor.distance = FLT_MAX;
				neighbor.cls = -1;
				neighbors.push_back(neighbor);
			}
		}

	}
}

__device__ int majorityVote(int k, Neighbor * neighbors) {
	struct FrequencyMap {
		int cls;
		int freq;
	};

	FrequencyMap * freqMap = (FrequencyMap *)malloc(sizeof(FrequencyMap)*k);

    int maxFrequency = 0;
    int  mostFrequentClass = neighbors[0].cls; //default, useful when k is 1
    int numClasses = 0;

	for(int i=0; i <k; i++) {
		bool found = false;
		for(int j=0; j < numClasses; j++) {
			if(freqMap[j].cls == neighbors[i].cls) {
				found = true;
				freqMap[j].freq = freqMap[j].freq + 1;
				if(freqMap[j].freq > maxFrequency) {
					maxFrequency = freqMap[j].freq;
					mostFrequentClass = freqMap[j].cls;
				}
				break;
			}
		}
		if(!found) {
			//Encountered this class first time. Add it to the map.
			freqMap[numClasses].cls = neighbors[i].cls;
			freqMap[numClasses].freq = 1;
			numClasses++;
		}
	}
	free(freqMap);
	return mostFrequentClass;
}


/*
 * Advance CUDA kernel function.
 * Each thread simply calculates the distance of one specific instance to another specific one.
 * Therefore, this model requires as many threads as the number of elements in the dataset
 */
__global__ void advanceCuda(Instance * instances, int numInstances, int numAttribs, float * distances)
{
	//First, compute the thread id and call it i.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= numInstances * numInstances) {
    	return;
    }

    Instance current_instance = instances[i/numInstances];
	Instance target_instance = instances[i%numInstances];

	float distance = 0;
	for(int h = 0; h < numAttribs; h++) // compute the distance between the two instances
	{
		float diff = current_instance.attribs[h] - target_instance.attribs[h];
		distance += diff * diff;
	}

	distance = sqrt(distance);

	distances[i] = distance;
}


/*
 * Basic CUDA kernel function.
 * Each threads runs KNN for exactly one instance in the dataset.
 * Therefore, this model requires as many threads as the number of elements in the dataset
 */
__global__ void basicCuda(Instance * instances, int numInstances, int numAttribs,
		int k, int * prediction)
{
	//First, compute the thread id and call it i.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= numInstances) {
    	return;
    }

    Instance current_instance = instances[i];

    //Array of k neighbors. Initialize them.
	Neighbor * neighbors = (Neighbor *)malloc(sizeof(Neighbor)*k);
	for(int p=0; p <k; p++) {
		neighbors[p].distance = FLT_MAX;
		neighbors[p].cls = -1;
	}


	for(int j = 0; j < numInstances; j++) // target each other instance
	{
		if(i == j) continue;

		float distance = 0;

		for(int h = 0; h < numAttribs; h++) // compute the distance between the two instances
		{
			float diff = current_instance.attribs[h] - instances[j].attribs[h];
			distance += diff * diff;
		}

		distance = sqrt(distance);

		for(int p=0; p <k; p++) {
			if(distance < neighbors[p].distance) {
				Neighbor neighbor;
				neighbor.distance = distance;
				neighbor.cls = instances[j].cls;

				Neighbor * newNeighbors = (Neighbor *)malloc(sizeof(Neighbor)*k);

				for(int q=0, r=0; q <k; q++) {
					if(p == q) {
						newNeighbors[q] = neighbor;
						continue;
					}
					newNeighbors[q] = neighbors[r++];
				}

				free(neighbors);
				neighbors = newNeighbors;
				break;
			}
		}
	}
	prediction[i] = majorityVote(k, neighbors);
	//Free the memory
	free(neighbors);
}

int* advanceCudaKNN(ArffData* dataset, int k)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();

    // Implement the KNN here, fill the predictions array
	cout << "K is " << k << endl;

	int numElements = dataset->num_instances();
	int numAttribs = dataset->num_attributes() - 1; //-1 because the last attrib is class

	Instance * h_instances = (Instance *)malloc(numElements * sizeof(Instance));

	// Launch the CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements * numElements / threadsPerBlock) + 1;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	//Convert the arf dataset to an array of Instance structure on host
	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		float * attribs = (float *) malloc (sizeof(float)*numAttribs);
		for(int h = 0; h < numAttribs; h++) // compute the distance between the two instances
		{
			attribs[h] = dataset->get_instance(i)->get(h)->operator float();
		}
		h_instances[i].attribs = attribs;
		h_instances[i].cls = dataset->get_instance(i)->get(numAttribs)->operator int32();
	}

	//Make another copy of the instances array from host to device
    Instance * d_instances;
	cudaMalloc(&d_instances, numElements*sizeof(Instance));
	cudaMemcpy(d_instances, h_instances, numElements*sizeof(Instance), cudaMemcpyHostToDevice);
	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		float * d_attribs;
		cudaMalloc(&d_attribs, numAttribs*sizeof(float));
		// Copy up attributes for each instance separately
		cudaMemcpy(d_attribs, h_instances[i].attribs, numAttribs*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_instances[i].attribs), &d_attribs, sizeof(float*), cudaMemcpyHostToDevice);
	}

	//Create an array of numElements X numElements elements on device. Kernel function will
	//populate the distances here.
	float * d_distances;
	cudaMalloc(&d_distances, numElements*numElements*sizeof(float));

	//Call kernel
	advanceCuda<<<blocksPerGrid, threadsPerBlock>>>(d_instances, numElements, numAttribs, d_distances);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

	//Create an array of numElements X numElements elements on host.
	//We will copy the distance array from the device to this one
	float * h_distances = (float *) malloc(numElements*numElements*sizeof(float));
	// Copy the device distance vector in device memory to the host distance vector
    cudaMemcpy(h_distances, d_distances, numElements * numElements * sizeof(float), cudaMemcpyDeviceToHost);

	predictFromDistances(h_distances, h_instances, numElements, k, predictions);

    // Free host memory
   	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		free(h_instances[i].attribs);
	}
    free(h_instances);
	free(h_distances);

    // Free device global memory
    cudaFree(d_distances);
	Instance * h_d_instances = (Instance *)malloc(numElements * sizeof(Instance));
	cudaMemcpy(h_d_instances, d_instances, numElements*sizeof(Instance), cudaMemcpyDeviceToHost);
	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		cudaFree(h_d_instances[i].attribs);
	}
    cudaFree(d_instances);
	free(h_d_instances);
	return predictions;
}

int* basicCudaKNN(ArffData* dataset, int k)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();

    // Implement the KNN here, fill the predictions array
	cout << "K is " << k << endl;

	int numElements = dataset->num_instances();
	int numAttribs = dataset->num_attributes() - 1; //-1 because the last attrib is class

	Instance * h_instances = (Instance *)malloc(numElements * sizeof(Instance));

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements / threadsPerBlock) + 1;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		float * attribs = (float *) malloc (sizeof(float)*numAttribs);
		for(int h = 0; h < numAttribs; h++) // compute the distance between the two instances
		{
			attribs[h] = dataset->get_instance(i)->get(h)->operator float();
		}
		h_instances[i].attribs = attribs;
		h_instances[i].cls = dataset->get_instance(i)->get(numAttribs)->operator int32();
	}

    Instance * d_instances;
	cudaMalloc(&d_instances, numElements*sizeof(Instance));
	cudaMemcpy(d_instances, h_instances, numElements*sizeof(Instance), cudaMemcpyHostToDevice);


	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		float * d_attribs;
		cudaMalloc(&d_attribs, numAttribs*sizeof(float));
		// Copy up attributes for each instance separately
		cudaMemcpy(d_attribs, h_instances[i].attribs, numAttribs*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_instances[i].attribs), &d_attribs, sizeof(float*), cudaMemcpyHostToDevice);
	}


	int * d_predictions;
	cudaMalloc(&d_predictions, numElements*sizeof(int));

	basicCuda<<<blocksPerGrid, threadsPerBlock>>>(d_instances, numElements,
			numAttribs, k, d_predictions);

	// Copy the device prediction vector in device memory to the host prediction vector
    cudaMemcpy(predictions, d_predictions, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Free host memory
   	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		free(h_instances[i].attribs);
	}
    free(h_instances);

    // Free device global memory
    cudaFree(d_predictions);

	Instance * h_d_instances = (Instance *)malloc(numElements * sizeof(Instance));
	cudaMemcpy(h_d_instances, d_instances, numElements*sizeof(Instance), cudaMemcpyDeviceToHost);
	for(int i = 0; i < numElements; i++) // for each instance in the dataset
	{
		cudaFree(h_d_instances[i].attribs);
	}
    cudaFree(d_instances);
	free(h_d_instances);
	return predictions;
}


int* KNN(ArffData* dataset, int k)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement the KNN here, fill the predictions array
	cout << "K is " << k << endl;
	
	for(int i = 0; i < dataset->num_instances(); i++) // for each instance in the dataset
	{
		//Initialize an empty neighbor
		Neighbor  neighbor;
		neighbor.distance = FLT_MAX;
		neighbor.cls = -1;
		
		//List of k neighbors
		std::list<Neighbor> neighbors (k, neighbor);
	
		for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
		{
			if(i == j) continue;
			
			float distance = 0;
			
			for(int h = 0; h < dataset->num_attributes() - 1; h++) // compute the distance between the two instances
			{
				float diff = dataset->get_instance(i)->get(h)->operator float() - dataset->get_instance(j)->get(h)->operator float();
				distance += diff * diff; 
			}
			
			distance = sqrt(distance);
			
			for (std::list<Neighbor>::iterator it = neighbors.begin(); it != neighbors.end(); it++) {
				if(distance < (*it).distance) {
					Neighbor neighbor;
					neighbor.distance = distance;
					neighbor.cls = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
					neighbors.insert(it, neighbor);
					neighbors.pop_back(); //Remove the last neighbor
					break;
				}
			}
		}
		
		predictions[i] = majorityVote(neighbors);
	}
	
    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./main <path to dataset> <k>" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    //int* predictions = KNN(dataset, atoi(argv[2]));
	//int* predictions = basicCudaKNN(dataset, atoi(argv[2]));
	int* predictions = advanceCudaKNN(dataset, atoi(argv[2]));

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
	
	//Free the memory
	free(predictions);
	free(confusionMatrix);
}
