public interface IClassifier<TIN, TOUT>
{
    /// <summary>
    /// Classifies the given testing features based on the training data.
    /// </summary>
    /// <param name="testFeatures">The features to classify.</param>
    /// <param name="trainFeatures">The features used for training.</param>
    /// <param name="trainLabels">The labels corresponding to the training features.</param>
    /// <param name="k">The number of neighbors to consider (if applicable).</param>
    /// <returns>The predicted labels for the test features.</returns>
    TOUT Classifier(TIN testingFeatures, TIN trainingFeatures, TOUT trainingLabels, int k);

    /// <summary>
    /// Calculates the accuracy of the predictions compared to the actual labels.
    /// </summary>
    /// <param name="predictedLabels">The labels predicted by the classifier.</param>
    /// <param name="actualLabels">The actual labels for comparison.</param>
    /// <returns>The accuracy of the predictions as a percentage.</returns>
    double CalculateAccuracy(TOUT predictedLabels, TOUT actualLabels);
}