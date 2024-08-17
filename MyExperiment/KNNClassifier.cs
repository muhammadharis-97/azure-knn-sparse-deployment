using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace KNNImplementation
{
    /// <summary>
    /// The KNNClassifier class implements the K-Nearest Neighbors algorithm to perform classification tasks on sequences derived from a Sparse Distributed Representation (SDR) dataset.
    /// </summary>
    public class KNNClassifier : IClassifier<List<List<double>>, List<string>>
    {
        /// <summary>
        /// Determines the class label by majority voting among the k nearest neighbors.
        /// </summary>
        /// <param name="nearestNeighbors">An array of IndexAndDistance objects representing the k nearest neighbors.</param>
        /// <param name="trainingLabels">The list of class labels corresponding to the training data.</param>
        /// <param name="k">The number of nearest neighbors to consider.</param>
        /// <returns>The class label with the most votes among the nearest neighbors.</returns>
        private int Vote(IndexAndDistance[] nearestNeighbors, List<string> trainingLabels, int k)
        {
            var votes = new Dictionary<string, int>();

            foreach (var label in trainingLabels)
            {
                if (!votes.ContainsKey(label))
                    votes[label] = 0;
            }

            for (int i = 0; i < k; i++)
            {
                string neighborLabel = trainingLabels[nearestNeighbors[i].idx];
                votes[neighborLabel]++;
            }

            // Find the class label with the most votes
            string classWithMostVotes = votes.OrderByDescending(pair => pair.Value).First().Key;
            return trainingLabels.IndexOf(classWithMostVotes);
        }

        /// <summary>
        /// Classifies the unknown SDRs based on the k-nearest neighbors in the training data using the KNN algorithm.
        /// </summary>
        /// <param name="testingFeatures">The features of the testing data to be classified.</param>
        /// <param name="trainingFeatures">The features of the training data used to classify the testing data.</param>
        /// <param name="trainingLabels">The class labels corresponding to the training data.</param>
        /// <param name="k">The number of nearest neighbors to consider in the classification.</param>
        /// <returns>A list of predicted labels for the testing features.</returns>
        public List<string> Classifier(List<List<double>> testingFeatures, List<List<double>> trainingFeatures, List<string> trainingLabels, int k)
        {
            Debug.WriteLine("Starting KNN Classification on Sparse Distributed Representations...");

            var predictedLabels = new List<string>();
            var calculateDistance = new DistanceCalculator();

            foreach (var testFeature in testingFeatures)
            {
                var nearestNeighbors = new IndexAndDistance[trainingFeatures.Count];
                for (int i = 0; i < trainingFeatures.Count; i++)
                {
                    double distance = calculateDistance.CalculateEuclideanDistance(testFeature, trainingFeatures[i]);
                    nearestNeighbors[i] = new IndexAndDistance { idx = i, dist = distance };
                }

                Array.Sort(nearestNeighbors);

                // Debug information for the k-nearest items
                Debug.WriteLine("Nearest Features / Euclidean Distance / Class Label");
                Debug.WriteLine("====================================================");
                for (int i = 0; i < k; i++)
                {
                    int nearestIndex = nearestNeighbors[i].idx;
                    double nearestDistance = nearestNeighbors[i].dist;
                    string nearestClass = trainingLabels[nearestIndex];
                    Debug.WriteLine($"({trainingFeatures[nearestIndex][0]}, {trainingFeatures[nearestIndex][1]}) : {nearestDistance} : {nearestClass}");
                }

                int resultIndex = Vote(nearestNeighbors, trainingLabels, k);
                predictedLabels.Add(trainingLabels[resultIndex]);
            }

            return predictedLabels;
        }

        /// <summary>
        /// Calculates the accuracy of the classifier by comparing predicted labels with actual labels.
        /// </summary>
        /// <param name="predictedLabels">The labels predicted by the classifier.</param>
        /// <param name="actualLabels">The actual labels from the testing dataset.</param>
        /// <returns>The accuracy of the classifier as a percentage.</returns>
        public double CalculateAccuracy(List<string> predictedLabels, List<string> actualLabels)
        {
            int correctPredictions = predictedLabels.Where((predictedLabel, index) => predictedLabel == actualLabels[index]).Count();
            return (double)correctPredictions / predictedLabels.Count * 100;
        }

        public List<SequenceDataEntry> LoadDataset(string datasetFilePath)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Represents an entry containing the index and distance for a training feature, used for comparison in the KNN algorithm.
    /// </summary>
    public class IndexAndDistance : IComparable<IndexAndDistance>
    {
        /// <summary>
        /// Index of the training feature.
        /// </summary>
        public int idx;

        /// <summary>
        /// Distance to the testing feature.
        /// </summary>
        public double dist;

        /// <summary>
        /// Compares this instance to another based on distance.
        /// </summary>
        public int CompareTo(IndexAndDistance other) => dist.CompareTo(other.dist);
    }

    /// <summary>
    /// Represents an entry in the dataset, containing a sequence name and its associated data.
    /// </summary>
    public class SequenceDataEntry
    {
        public string SequenceName { get; set; }
        public List<double> SequenceData { get; set; }
    }

    /// <summary>
    /// Provides methods for calculating various distances between features, used in the KNN algorithm.
    /// </summary>
    public class DistanceCalculator
    {
        /// <summary>
        /// Calculates the Euclidean distance between two features.
        /// </summary>
        /// <param name="testFeature">Feature from the testing data.</param>
        /// <param name="trainFeature">Feature from the training data.</param>
        /// <returns>The Euclidean distance between the two features.</returns>
        public double CalculateEuclideanDistance(List<double> testFeature, List<double> trainFeature)
        {
            ValidateFeatureData(testFeature, trainFeature);
            return Math.Sqrt(testFeature.Zip(trainFeature, (test, train) => Math.Pow(test - train, 2)).Sum());
        }

        /// <summary>
        /// Calculates the Manhattan distance between two features.
        /// </summary>
        /// <param name="testFeature">Feature from the testing data.</param>
        /// <param name="trainFeature">Feature from the training data.</param>
        /// <returns>The Manhattan distance between the two features.</returns>
        public double CalculateManhattanDistance(List<double> testFeature, List<double> trainFeature)
        {
            ValidateFeatureData(testFeature, trainFeature);
            return testFeature.Zip(trainFeature, (test, train) => Math.Abs(test - train)).Sum();
        }

        /// <summary>
        /// Calculates the Minkowski distance between two features.
        /// </summary>
        /// <param name="testFeature">Feature from the testing data.</param>
        /// <param name="trainFeature">Feature from the training data.</param>
        /// <param name="p">The order parameter of the Minkowski distance.</param>
        /// <returns>The Minkowski distance between the two features.</returns>
        public double CalculateMinkowskiDistance(List<double> testFeature, List<double> trainFeature, int p)
        {
            ValidateFeatureData(testFeature, trainFeature);
            return Math.Pow(testFeature.Zip(trainFeature, (test, train) => Math.Pow(Math.Abs(test - train), p)).Sum(), 1.0 / p);
        }

        /// <summary>
        /// Validates that both feature sets are non-null and of equal length.
        /// </summary>
        private void ValidateFeatureData(List<double> testFeature, List<double> trainFeature)
        {
            if (testFeature == null || trainFeature == null)
                throw new ArgumentNullException("Both testFeature and trainFeature must not be null.");

            if (testFeature.Count != trainFeature.Count)
                throw new ArgumentException("testFeature and trainFeature must have the same length.");
        }
    }
}