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
		float smallestDistance = FLT_MAX;
		int smallestDistanceClass;

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
        cout << "Usage: ./main ../datasets/datasetFile.arff <k>" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    int* predictions = KNN(dataset, atoi(argv[2]));
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
