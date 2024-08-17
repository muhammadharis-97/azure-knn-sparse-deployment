using Azure.Storage.Queues;
using Azure.Storage.Queues.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using MyCloudProject.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using OfficeOpenXml;
using Newtonsoft.Json;
using KNNImplementation;

namespace MyExperiment
{

    /// <summary>
    /// Initializes a new instance of the <see cref="Experiment"/> class.
    /// </summary>
    /// <param name="configSection">The configuration section to bind to the <see cref="MyConfig"/> object.</param>
    /// <param name="storageProvider">An instance of <see cref="IStorageProvider"/> for handling storage operations.</param>
    /// <param name="log">An instance of <see cref="ILogger"/> used for logging operations.</param>
    public class Experiment : IExperiment
    {

        private readonly IStorageProvider _storageProvider;
        private readonly ILogger _logger;
        private readonly MyConfig _config;

        // Initialize and bind configuration settings from the provided configuration section
        public Experiment(IConfigurationSection configSection, IStorageProvider storageProvider, ILogger log)
        {
            _storageProvider = storageProvider;
            _logger = log;
            _config = new MyConfig();
            configSection.Bind(_config);
        }

        /// <summary>
        /// Executes the ML experiment by processing each testing file against the training data.
        /// </summary>
        /// <param name="trainingDataFolder">The path to the folder containing training data files.</param>
        /// <param name="testingDataFolder">The path to the folder containing testing data files.</param>
        /// <returns>A list of experiment results for each testing file processed.</returns>
        public async Task<List<IExperimentResult>> RunAsync(string trainingDataFolder, string testingDataFolder)
        {
            var sequences = ReadSequencesJson();
            var paths = CreateOutputFolderAndFile();

            if (string.IsNullOrEmpty(paths.filePath))
            {
                throw new InvalidOperationException("The output file path is not correctly initialized.");
            }

            string excelName = "KNN_Results.xlsx";
            string excelFilePath = Path.Combine(Directory.GetCurrentDirectory(), excelName);

            _logger.LogInformation($"Excel file path: {excelFilePath}");

            List<IExperimentResult> results = new List<IExperimentResult>();

            try
            {
                DeleteExistingFile(excelFilePath);
                CreateNewExcelFile(excelFilePath);
                _logger.LogInformation($"File {excelName} created successfully.");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error handling Excel file: {ex.Message}");
            }

            foreach (string testingFile in Directory.GetFiles(testingDataFolder))
            {
                var result = new ExperimentResult(this._config.GroupId, "303");

                try
                {
                    _logger.LogInformation($"{DateTime.UtcNow} - Processing testing file: {testingFile} with training data in folder: {trainingDataFolder}");

                    result.StartTimeUtc = DateTime.UtcNow;

                    var knnResult = KNNClassificationExperiment(trainingDataFolder, testingFile);

                    string outputFileName = Path.Combine(paths.folderPath, $"{Path.GetFileNameWithoutExtension(testingFile)}_output.txt");
                    await File.WriteAllLinesAsync(outputFileName, knnResult.PredictedLabels);

                    Console.WriteLine("Predicted Labels:");
                    knnResult.PredictedLabels.ForEach(label => Console.WriteLine(label));
                    Console.WriteLine($"Accuracy: {knnResult.Accuracy}%");

                    result.EndTimeUtc = DateTime.UtcNow;
                    result.DurationSec = (long)(result.EndTimeUtc - result.StartTimeUtc)?.TotalSeconds;
                    result.OutputFilesProxy = new[] { outputFileName };
                    result.TrainingFileUrl = trainingDataFolder;  // Refers to the folder
                    result.TestingFileUrl = testingFile;
                    result.predictedLabels = knnResult.PredictedLabels;
                    result.Accuracy = (float)knnResult.Accuracy;

                    _logger.LogInformation($"Timestamp: {result.Timestamp}");
                    _logger.LogInformation($"EndTimeUtc: {result.EndTimeUtc}");
                    _logger.LogInformation($"DurationSec: {result.DurationSec}");
                    _logger.LogInformation($"OutputFilesProxy: {string.Join(", ", result.OutputFilesProxy)}");
                    _logger.LogInformation($"Predicted labels: {string.Join(", ", result.predictedLabels)}");
                    _logger.LogInformation($"Accuracy: {result.Accuracy}");

                    foreach (var (actualLabel, predictedLabel) in knnResult.PredictedLabels.Zip(knnResult.ActualLabels))
                    {
                        var actualSequence = string.Join(", ", sequences[actualLabel]);
                        var predictedSequence = string.Join(", ", sequences[predictedLabel]);

                        var logMessage = $"Actual Label: {actualLabel}, Predicted Label: {predictedLabel}\n" +
                                         $"Actual Sequence: [{actualSequence}]\n" +
                                         $"Predicted Sequence: [{predictedSequence}]";

                        _logger.LogInformation(logMessage);
                    }

                    SaveResultsToExcel(result, excelFilePath);
                    SaveResultsToOutputFile(result, outputFileName);

                    results.Add(result);

                    _logger.LogInformation($"{DateTime.UtcNow} - Completed processing file: {testingFile}");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error processing testing file: {testingFile} with training data in folder: {trainingDataFolder}");
                }
            }

            return results;
        }

        /// <summary>
        /// Creates an output folder and an output file for storing results. 
        /// If the output folder already exists, it is deleted and recreated.
        /// </summary>
        /// <returns>A tuple containing the paths to the output folder and the output file.</returns>
        private static (string folderPath, string filePath) CreateOutputFolderAndFile()
        {
            // Define the path for the output folder and the output file
            string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "OutputFolder");
            string outputFile = Path.Combine(outputFolder, "output.txt");

            try
            {
                // Check if the output folder already exists
                if (Directory.Exists(outputFolder))
                {
                    Console.WriteLine("Output folder exists. Deleting...");
                    // Delete the existing output folder and its contents
                    Directory.Delete(outputFolder, true);
                    Console.WriteLine("Output folder deleted successfully.");
                }

                // Create a new output folder
                Directory.CreateDirectory(outputFolder);
                Console.WriteLine("Output folder created successfully.");

                // Create and write to the output text file
                using (StreamWriter writer = new StreamWriter(outputFile))
                {
                    writer.WriteLine("This is the output text file.");
                }
                Console.WriteLine("Output file created successfully at: " + outputFile);

                // Return the paths to the output folder and the output file
                return (outputFolder, outputFile);
            }
            catch (UnauthorizedAccessException ex)
            {
                // Handle unauthorized access errors (e.g., insufficient permissions)
                Console.WriteLine("Error: Unauthorized access. Please check your permissions.");
                Console.WriteLine(ex.Message);
                throw;
            }
            catch (IOException ex)
            {
                // Handle I/O errors (e.g., file system issues)
                Console.WriteLine("Error: An I/O error occurred.");
                Console.WriteLine(ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                // Handle any other unexpected errors
                Console.WriteLine("Error: An unexpected error occurred.");
                Console.WriteLine(ex.Message);
                throw;
            }
        }


        /// <summary>
        /// Deletes a specified file if it exists.
        /// </summary>
        /// <param name="filePath">The full path to the file to be deleted.</param> 
        /// <exception cref="Exception">Throws an exception if an error occurs while deleting the file.</exception>
        private static void DeleteExistingFile(string filePath)
        {
            // Check if the file exists at the specified path
            if (File.Exists(filePath))
            {
                try
                {
                    // Attempt to delete the file
                    File.Delete(filePath);
                    Console.WriteLine($"File {Path.GetFileName(filePath)} deleted successfully.");
                }
                catch (Exception ex)
                {
                    // Log the error message if an exception occurs during file deletion
                    Console.Error.WriteLine($"Error deleting file {Path.GetFileName(filePath)}: {ex.Message}");
                    // Rethrow the exception to propagate the error
                    throw;
                }
            }
            else
            {
                // Inform the user that the file does not exist
                Console.WriteLine($"File {Path.GetFileName(filePath)} does not exist.");
            }
        }

        /// <summary>
        /// Creates a new Excel file at the specified path. If the directory does not exist, it will be created. 
        /// An Excel worksheet named "Sheet1" is added to the file.
        /// </summary>
        /// <param name="filePath">The full path where the new Excel file will be created.</param>
        /// <exception cref="ArgumentException">Thrown if the provided file path is null or empty.</exception>
        /// <exception cref="IOException">Thrown if the file is locked or cannot be accessed.</exception>
        private static void CreateNewExcelFile(string filePath)
        {
            // Set the license context for the ExcelPackage to non-commercial use
            OfficeOpenXml.ExcelPackage.LicenseContext = OfficeOpenXml.LicenseContext.NonCommercial;

            // Validate that the file path is not null or empty
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
            }

            // Ensure the directory for the file exists; if not, create it
            string directoryPath = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            try
            {
                // Create a FileInfo object for the specified file path
                var fileInfo = new FileInfo(filePath);

                // Check if the file already exists and if it is locked
                if (fileInfo.Exists && IsFileLocked(fileInfo))
                {
                    throw new IOException($"The file {filePath} is currently in use by another process.");
                }

                // Create a new Excel package
                using (var package = new ExcelPackage(fileInfo))
                {
                    // Add a worksheet named "Sheet1"
                    var worksheet = package.Workbook.Worksheets.Add("Sheet1");

                    // Save the package to the specified file path
                    package.Save();
                }

                // Log the successful creation of the Excel file
                Console.WriteLine($"Excel file created successfully at {filePath}");
            }
            catch (Exception ex)
            {
                // Log the error and rethrow the exception
                Console.Error.WriteLine($"Error creating Excel file {Path.GetFileName(filePath)}: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Determines if the specified file is currently locked by another process.
        /// </summary>
        /// <param name="file">The <see cref="FileInfo"/> object representing the file to check.</param>
        /// <returns>Returns <c>true</c> if the file is locked or otherwise unavailable; otherwise, returns <c>false</c>.</returns>
        /// <remarks>
        /// This method attempts to open the file with exclusive access. If the file is locked by another process, 
        /// an <see cref="IOException"/> will be thrown, indicating that the file is currently in use or unavailable.
        /// </remarks>
        private static bool IsFileLocked(FileInfo file)
        {
            try
            {
                // Attempt to open the file with exclusive read-write access
                using (FileStream stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                {
                    // Close the stream if the file is successfully opened
                    stream.Close();
                }
            }
            catch (IOException)
            {
                // An IOException indicates that the file is locked by another process or is otherwise unavailable
                return true;
            }

            // File is not locked; it can be accessed
            return false;
        }
        /// <summary>
        /// Saves the experiment results to the specified output file.
        /// </summary>
        /// <param name="result">The <see cref="ExperimentResult"/> object containing the experiment results to be saved.</param>
        /// <param name="outputFilePath">The path to the output file where the results will be saved.</param>
        /// <remarks>
        /// This method formats the experiment results into a list of strings and writes them to a file.
        /// It logs the success or failure of the operation and includes detailed error handling.
        /// </remarks>
        private void SaveResultsToOutputFile(ExperimentResult result, string outputFilePath)
        {
            try
            {
                // Prepare the content to be written to the file
                var content = new List<string>
            {
                $"Experiment ID: {result.ExperimentId}",
                $"Timestamp: {result.Timestamp:yyyy-MM-dd HH:mm:ss}",
                $"Start Time (UTC): {result.StartTimeUtc:yyyy-MM-dd HH:mm:ss}",
                $"End Time (UTC): {result.EndTimeUtc:yyyy-MM-dd HH:mm:ss}",
                $"Duration (seconds): {result.DurationSec}",
                $"Training Data Directory: {result.TrainingFileUrl}",
                $"Testing File: {result.TestingFileUrl}",
                $"Accuracy: {result.Accuracy:F2}%" // Format accuracy to 2 decimal places
            };

                // Add predicted labels with indexing for clarity
                content.Add("Predicted Labels:");
                content.AddRange(result.predictedLabels.Select((label, index) => $"{index + 1}: {label}"));

                // Write the content to the output file
                File.WriteAllLines(outputFilePath, content);

                // Log the success message
                _logger?.LogInformation($"Results successfully written to: {outputFilePath}");
            }
            catch (Exception ex)
            {
                // Log the error message and throw an exception with detailed information
                _logger?.LogError($"Failed to save results to output file at '{outputFilePath}': {ex.Message}");
                throw new InvalidOperationException($"An error occurred while writing results to the output file: {outputFilePath}", ex);
            }
        }

        /// <summary>
        /// Saves the experiment results to an Excel file, creating or updating the file as needed.
        /// </summary>
        /// <param name="result">The <see cref="ExperimentResult"/> object containing the experiment results to be saved.</param>
        /// <param name="excelFileName">The name of the Excel file to save the results to.</param>
        /// <remarks>
        /// This method creates or updates an Excel file with the experiment results. It checks if the file exists,
        /// loads it if necessary, and appends the new results to the appropriate worksheet. Headers are set if the 
        /// worksheet is newly created. It also handles any potential errors during file operations.
        /// </remarks>
        public static void SaveResultsToExcel(ExperimentResult result, string excellfilename)
        {
            string excelFilePath = Path.Combine(Directory.GetCurrentDirectory(), excellfilename);

            try
            {
                using (var package = new ExcelPackage())
                {
                    // Load existing Excel file if it exists
                    if (File.Exists(excelFilePath))
                    {
                        var existingFile = new FileInfo(excelFilePath);
                        using (var stream = existingFile.Open(FileMode.Open, FileAccess.ReadWrite))
                        {
                            package.Load(stream);
                        }
                    }

                    // Add or get the worksheet
                    var worksheet = package.Workbook.Worksheets.FirstOrDefault(ws => ws.Name == "Experiment Result");
                    if (worksheet == null)
                    {
                        worksheet = package.Workbook.Worksheets.Add("Experiment Result");

                        // Set headers
                        worksheet.Cells[1, 1].Value = "Timestamp";
                        worksheet.Cells[1, 2].Value = "EndTimeUtc";
                        worksheet.Cells[1, 3].Value = "ExperimentId";
                        worksheet.Cells[1, 4].Value = "DurationSec";
                        worksheet.Cells[1, 5].Value = "OutputFilesProxy";
                        worksheet.Cells[1, 6].Value = "TrainingFileUrl";
                        worksheet.Cells[1, 7].Value = "TestingFileUrl";
                        worksheet.Cells[1, 8].Value = "predictedLabels";
                        worksheet.Cells[1, 9].Value = "Accuracy";
                    }

                    // Find the next row to append data
                    int nextRow = worksheet.Dimension?.Rows + 1 ?? 2;

                    // Populate data row
                    worksheet.Cells[nextRow, 1].Value = result.Timestamp.ToString();
                    worksheet.Cells[nextRow, 2].Value = result.EndTimeUtc.ToString();
                    worksheet.Cells[nextRow, 3].Value = result.ExperimentId;
                    worksheet.Cells[nextRow, 4].Value = result.DurationSec;
                    worksheet.Cells[nextRow, 5].Value = string.Join(", ", result.OutputFilesProxy);
                    worksheet.Cells[nextRow, 6].Value = result.TrainingFileUrl;
                    worksheet.Cells[nextRow, 7].Value = result.TestingFileUrl;
                    worksheet.Cells[nextRow, 8].Value = result.predictedLabels;
                    worksheet.Cells[nextRow, 8].Value = result.Accuracy;

                    // Auto fit columns for better readability
                    worksheet.Cells.AutoFitColumns();

                    // Save Excel file to disk
                    package.SaveAs(new FileInfo(excelFilePath));
                }
            }
            catch (Exception ex)
            {
                // Log or handle the exception as needed
                Console.Error.WriteLine($"Error saving results to Excel file: {ex.Message}");
            }
        }


        /// <summary>
        /// Reads and deserializes a JSON file into a dictionary of sequences.
        /// </summary>
        /// <param name="fileName">The name of the JSON file containing the sequences (default is "sequences.json").</param>
        /// <returns>A dictionary where the keys are sequence identifiers and the values are lists of doubles representing the sequences.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the specified file does not exist.</exception>
        /// <exception cref="JsonSerializationException">Thrown when the JSON data cannot be deserialized into the expected format.</exception>
        /// <exception cref="IOException">Thrown when an I/O error occurs while reading the file.</exception>
        public Dictionary<string, List<double>> ReadSequencesJson(string fileName = "sequences.json")
        {
            // Combine the base directory with the filename to get the full path
            string jsonFilePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
            Console.WriteLine($"Current Directory: {Directory.GetCurrentDirectory()}");

            // Check if the file exists
            if (!File.Exists(jsonFilePath))
            {
                throw new FileNotFoundException($"The file '{fileName}' was not found at path '{jsonFilePath}'.");
            }

            try
            {
                // Read the JSON content from the file
                string jsonData = File.ReadAllText(jsonFilePath);

                // Deserialize the JSON content into a dictionary
                var sequences = JsonConvert.DeserializeObject<Dictionary<string, List<double>>>(jsonData);

                if (sequences == null)
                {
                    throw new JsonSerializationException("The JSON data could not be deserialized into the expected format.");
                }

                return sequences;
            }
            catch (JsonSerializationException ex)
            {
                // Handle JSON deserialization errors
                Console.Error.WriteLine($"Error deserializing JSON data from file '{fileName}': {ex.Message}");
                throw;
            }
            catch (IOException ex)
            {
                // Handle I/O errors
                Console.Error.WriteLine($"I/O error reading file '{fileName}': {ex.Message}");
                throw;
            }
            catch (Exception ex)
            {
                // Handle any other unexpected errors
                Console.Error.WriteLine($"Unexpected error occurred: {ex.Message}");
                throw;
            }
        }


        /// <summary>
        /// Represents the result of a K-Nearest Neighbors (KNN) classification experiment.
        /// </summary>
        public class KNNClassificationResult
        {
            /// <summary>
            /// Gets or sets the list of predicted labels for the test data.
            /// </summary>
            public List<string> PredictedLabels { get; set; }

            /// <summary>
            /// Gets or sets the list of actual labels for the test data.
            /// </summary>
            public List<string> ActualLabels { get; set; }

            /// <summary>
            /// Gets or sets the classification accuracy of the KNN model.
            /// </summary>
            public double Accuracy { get; set; }
        }

        /// <summary>
        /// Performs a K-Nearest Neighbors (KNN) classification experiment using the specified training and testing data.
        /// </summary>
        /// <param name="trainingFolder">The directory path containing the training data files in JSON format.</param>
        /// <param name="testingFile">The path to the testing data file in JSON format.</param>
        /// <returns>A <see cref="KNNClassificationResult"/> object containing the results of the classification experiment.</returns>
        /// <exception cref="DirectoryNotFoundException">Thrown if the specified training folder does not exist.</exception>
        /// <exception cref="FileNotFoundException">Thrown if the specified testing file does not exist.</exception>
        public static KNNClassificationResult KNNClassificationExperiment(string trainingFolder, string testingfile)
        {
            // Instantiate the Classifierleaning and KNNClassifier classes
            Classifierleaning classifierleaning = new Classifierleaning();
            KNNClassifier kNNClassifier = new KNNClassifier();

            List<SequenceDataEntry> sequenceDataEntriestraining = new List<SequenceDataEntry>();
            List<SequenceDataEntry> sequenceDataEntriestesting = new List<SequenceDataEntry>();

            // Load all files from the training folder and add to the training data list
            foreach (string trainingfile in Directory.GetFiles(trainingFolder, "*.json"))
            {
                sequenceDataEntriestraining.AddRange(classifierleaning.LoadDataset(trainingfile));
            }

            // Load the testing file
            sequenceDataEntriestesting.AddRange(classifierleaning.LoadDataset(testingfile));

            // Extract the training data labels and features.
            Classifierleaning.ExtractFeaturesAndLabels(sequenceDataEntriestraining, out List<List<double>> trainingFeatures, out List<string> trainingLabels);

            // Extract the testing data labels and features.
            Classifierleaning.ExtractFeaturesAndLabels(sequenceDataEntriestesting, out List<List<double>> testingFeatures, out List<string> testingLabels);

            // Perform classification on the testing set using the KNN algorithm
            List<string> predictedLabels = kNNClassifier.Classifier(testingFeatures, trainingFeatures, trainingLabels, k: 3);

            // Calculate the accuracy of the classifier
            double accuracy = kNNClassifier.CalculateAccuracy(predictedLabels, testingLabels);

            // Return the result
            return new KNNClassificationResult
            {
                PredictedLabels = predictedLabels,
                ActualLabels = testingLabels,  // Populate the actual labels
                Accuracy = accuracy
            };
        }

    }
}