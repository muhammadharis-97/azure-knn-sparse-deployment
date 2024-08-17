
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace KNNImplementation
{
    /// <summary>
    /// Utility class for loading datasets and extracting features and labels.
    /// </summary>
    public class Classifierleaning
    {
        /// <summary>
        /// Loads a dataset from a JSON file and returns a list of sequence data entries.
        /// </summary>
        /// <param name="filePath">The file path to the JSON dataset.</param>
        /// <returns>A list of <see cref="SequenceDataEntry"/> objects representing the dataset.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the specified file is not found.</exception>
        /// <exception cref="JsonException">Thrown when there is an error in deserializing the JSON data.</exception>
        public List<SequenceDataEntry> LoadDataset(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException("The specified file was not found.", filePath);
            }

            string jsonData = File.ReadAllText(filePath);
            try
            {
                return JsonConvert.DeserializeObject<List<SequenceDataEntry>>(jsonData)
                       ?? new List<SequenceDataEntry>();
            }
            catch (JsonException ex)
            {
                throw new JsonException("Error deserializing the JSON data.", ex);
            }
        }

        /// <summary>
        /// Extracts features and labels from a list of sequence data entries.
        /// </summary>
        /// <param name="sequenceDataEntries">The list of sequence data entries.</param>
        /// <param name="features">The list to hold extracted features.</param>
        /// <param name="labels">The list to hold extracted labels.</param>
        public static void ExtractFeaturesAndLabels(List<SequenceDataEntry> sequenceDataEntries,
            out List<List<double>> features, out List<string> labels)
        {
            if (sequenceDataEntries == null) throw new ArgumentNullException(nameof(sequenceDataEntries));

            features = new List<List<double>>();
            labels = new List<string>();

            foreach (var entry in sequenceDataEntries)
            {
                if (entry == null) continue;

                features.Add(new List<double>(entry.SequenceData));
                labels.Add(entry.SequenceName);
            }
        }


        /// <summary>
        /// Splits the dataset into training and testing data.
        /// </summary>
        /// <param name="sequenceDataEntries">The list of sequence data entries.</param>
        /// <param name="trainingFeatures">The list to hold training features.</param>
        /// <param name="testingFeatures">The list to hold testing features.</param>
        /// <param name="trainingLabels">The list to hold training labels.</param>
        /// <param name="testingLabels">The list to hold testing labels.</param>
        public static void SplitDataset(List<SequenceDataEntry> sequenceDataEntries,
            out List<List<double>> trainingFeatures, out List<List<double>> testingFeatures,
            out List<string> trainingLabels, out List<string> testingLabels)
        {
            if (sequenceDataEntries == null) throw new ArgumentNullException(nameof(sequenceDataEntries));

            // Assuming a 70-30 split for training and testing data
            int trainingDataCount = (int)(sequenceDataEntries.Count * 0.7);

            trainingFeatures = new List<List<double>>();
            testingFeatures = new List<List<double>>();
            trainingLabels = new List<string>();
            testingLabels = new List<string>();

            for (int i = 0; i < sequenceDataEntries.Count; i++)
            {
                if (i < trainingDataCount)
                {
                    trainingFeatures.Add(new List<double>(sequenceDataEntries[i].SequenceData));
                    trainingLabels.Add(sequenceDataEntries[i].SequenceName);
                }
                else
                {
                    testingFeatures.Add(new List<double>(sequenceDataEntries[i].SequenceData));
                    testingLabels.Add(sequenceDataEntries[i].SequenceName);
                }
            }
        }
    }
}
