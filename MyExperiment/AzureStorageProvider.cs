using Azure;
using Azure.Data.Tables;
using Azure.Storage.Blobs;
using Azure.Storage.Queues;
using Azure.Storage.Queues.Models;
using KNNImplementation;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using MyCloudProject.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace MyExperiment
{
    public class AzureStorageProvider : IStorageProvider
    {
        private MyConfig _config;
        private readonly ILogger _logger;

        public AzureStorageProvider(IConfigurationSection configSection, ILogger log)
        {
            _config = new MyConfig();
            _logger = log;
            configSection.Bind(_config);
        }

        /// <summary>
        /// Deletes a message from the Azure Queue based on the provided experiment request.
        /// </summary>
        /// <param name="request">The experiment request containing message details.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task CommitRequestAsync(IExperimentRequest request)
        {
            if (request == null)
            {
                throw new ArgumentNullException(nameof(request), "The experiment request cannot be null.");
            }

            try
            {
                // Initialize the queue client using the storage connection string and queue name
                var queueClient = new QueueClient(this._config.StorageConnectionString, this._config.Queue);

                // Delete the message from the queue using the MessageId and message receipt
                await queueClient.DeleteMessageAsync(request.MessageId, request.PopReceipt);

                // Log the successful deletion
                _logger?.LogInformation($"Message with ID {request.MessageId} deleted successfully.");
            }
            catch (ArgumentNullException ex)
            {
                // Handle specific exception when argument is null
                _logger?.LogError($"Argument null exception: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
            catch (ArgumentException ex)
            {
                // Handle specific exception when argument is invalid
                _logger?.LogError($"Argument exception: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
            catch (RequestFailedException ex)
            {
                // Handle Azure SDK request failed exception
                _logger?.LogError($"Failed to delete message: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
            catch (Exception ex)
            {
                // Handle any other unexpected exceptions
                _logger?.LogError($"An error occurred: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
        }


        /// <summary>
        /// Downloads input files from Azure Blob Storage based on the provided file names and experiment ID.
        /// </summary>
        /// <param name="trainingFileName">The name of the training file to download.</param>
        /// <param name="testingFileName">The name of the testing file to download.</param>
        /// <param name="experimentId">The experiment ID indicating which files to download.</param>
        /// <returns>The local folder path where the files are downloaded.</returns>

        public async Task<LocalFileWithInputArgs> DownloadInputAsync(string trainingFileName, string testingFileName, string experimentId)
        {
            if (string.IsNullOrWhiteSpace(trainingFileName) || string.IsNullOrWhiteSpace(testingFileName))
            {
                throw new ArgumentException("File names cannot be null or whitespace.");
            }

            if (string.IsNullOrWhiteSpace(experimentId))
            {
                throw new ArgumentException("Experiment ID cannot be null or whitespace.", nameof(experimentId));
            }

            // Separate container clients for training and testing data
            var trainingContainerClient = new BlobContainerClient(_config.StorageConnectionString, _config.TrainingContainer);
            var testingContainerClient = new BlobContainerClient(_config.StorageConnectionString, _config.TestingContainer);

            await trainingContainerClient.CreateIfNotExistsAsync();
            await testingContainerClient.CreateIfNotExistsAsync();

            var downloadFolderPathTraining = Path.Combine(Directory.GetCurrentDirectory(), "TrainingData");
            var downloadFolderPathTesting = Path.Combine(Directory.GetCurrentDirectory(), "TestingData");

            // Clean up the download folders before starting
            if (Directory.Exists(downloadFolderPathTraining))
            {
                Directory.Delete(downloadFolderPathTraining, true);
            }
            Directory.CreateDirectory(downloadFolderPathTraining);

            if (Directory.Exists(downloadFolderPathTesting))
            {
                Directory.Delete(downloadFolderPathTesting, true);
            }
            Directory.CreateDirectory(downloadFolderPathTesting);

            string trainingDownloadFilePath = null;
            string testingDownloadFilePath = null;

            try
            {
                switch (experimentId)
                {
                    case "2":
                        _logger?.LogInformation($"{DateTime.Now} - ExperimentId is 2, downloading all files from the containers.");

                        // Download all files from the training data container
                        await foreach (var blobItem in trainingContainerClient.GetBlobsAsync())
                        {
                            var blobClient = trainingContainerClient.GetBlobClient(blobItem.Name);
                            trainingDownloadFilePath = Path.Combine(downloadFolderPathTraining, blobItem.Name);
                            await blobClient.DownloadToAsync(trainingDownloadFilePath);
                            _logger?.LogInformation($"{DateTime.Now} - Downloaded training file: {trainingDownloadFilePath}");
                        }

                        // Download all files from the testing data container
                        await foreach (var blobItem in testingContainerClient.GetBlobsAsync())
                        {
                            var blobClient = testingContainerClient.GetBlobClient(blobItem.Name);
                            testingDownloadFilePath = Path.Combine(downloadFolderPathTesting, blobItem.Name);
                            await blobClient.DownloadToAsync(testingDownloadFilePath);
                            _logger?.LogInformation($"{DateTime.Now} - Downloaded testing file: {testingDownloadFilePath}");
                        }
                        break;

                    case "1":
                        _logger?.LogInformation($"{DateTime.Now} - ExperimentId is 1, downloading the specified files: {trainingFileName} and {testingFileName}");

                        // Download all files from the training data container
                        await foreach (var blobItem in trainingContainerClient.GetBlobsAsync())
                        {
                            var blobClient = trainingContainerClient.GetBlobClient(blobItem.Name);
                            trainingDownloadFilePath = Path.Combine(downloadFolderPathTraining, blobItem.Name);
                            await blobClient.DownloadToAsync(trainingDownloadFilePath);
                            _logger?.LogInformation($"{DateTime.Now} - Downloaded training file: {trainingDownloadFilePath}");
                        }

                        // Download specified testing file
                        var testingBlobClient = testingContainerClient.GetBlobClient(testingFileName);
                        testingDownloadFilePath = Path.Combine(downloadFolderPathTesting, testingFileName);
                        await testingBlobClient.DownloadToAsync(testingDownloadFilePath);

                        _logger?.LogInformation($"{DateTime.Now} - Downloaded testing file: {testingDownloadFilePath}");
                        break;

                    default:
                        _logger?.LogWarning($"{DateTime.Now} - ExperimentId is not recognized. No files downloaded.");
                        break;
                }
            }
            catch (RequestFailedException ex)
            {
                _logger?.LogError($"Failed to download files from Azure Blob Storage: {ex.Message}");
                throw;
            }
            catch (Exception ex)
            {
                _logger?.LogError($"An error occurred while downloading files: {ex.Message}");
                throw;
            }

            return new LocalFileWithInputArgs
            {
                TrainingDataFolder = downloadFolderPathTraining,
                TestingDataFolder = downloadFolderPathTesting
            };
        }



        /// <summary>
        /// Receives an experiment request message from an Azure Queue.
        /// </summary>
        /// <param name="token">Cancellation token to stop receiving messages.</param>
        /// <returns>An instance of IExperimentRequest representing the received experiment request, or null if no message is received.</returns>
        public async Task<IExperimentRequest> ReceiveExperimentRequestAsync(CancellationToken token)
        {
            // Initialize the QueueClient with the storage connection string and queue name
            QueueClient queueClient = new QueueClient(this._config.StorageConnectionString, this._config.Queue);

            // Continuously attempt to receive messages until cancellation is requested
            while (!token.IsCancellationRequested)
            {
                // Receive messages from the queue asynchronously
                QueueMessage[] messages = await queueClient.ReceiveMessagesAsync();

                // Check if any messages were received
                if (messages != null && messages.Length > 0)
                {
                    foreach (var message in messages)
                    {
                        try
                        {
                            // Extract message text from the message body
                            string msgTxt = message.Body.ToString();

                            // Output the message text to the console (optional)
                            await Console.Out.WriteLineAsync(msgTxt);

                            // Log that a message has been received
                            _logger?.LogInformation($"{DateTime.Now} - Received the message: {msgTxt}");

                            var data = Convert.FromBase64String(msgTxt);
                            var jsonString = Encoding.UTF8.GetString(data);

                            // Deserialize the message JSON into an ExerimentRequestMessage object
                            var request = JsonSerializer.Deserialize<ExperimentRequestMessage>(jsonString);

                            // Check if deserialization was successful
                            if (request != null)
                            {
                                // Assign message ID and pop receipt to the request object
                                request.MessageId = message.MessageId;
                                request.PopReceipt = message.PopReceipt;

                                // Output the input file path to the console (optional)
                                await Console.Out.WriteLineAsync(request.TrainingDataFile);
                                await Console.Out.WriteLineAsync(request.TestingDataFile);

                                // Return the deserialized request object
                                return request;
                            }
                        }
                        catch (JsonException ex)
                        {
                            // Log an error if deserialization fails
                            _logger?.LogError($"{DateTime.Now} - Failed to deserialize the message. Exception: {ex.Message}");
                        }
                        catch (Exception ex)
                        {
                            // Log an error for any other exceptions during message processing
                            _logger?.LogError($"{DateTime.Now} - An error occurred while processing the message. Exception: {ex.Message}");
                        }
                    }
                }

                // Optional delay before retrying to receive messages
                await Task.Delay(1000);
            }

            // Return null if cancellation is requested or no messages are received
            return null;
        }


        /// <summary>
        /// Uploads experiment results to Azure Blob Storage.
        /// </summary>
        /// <param name="folderPath">The local folder path containing files to upload.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task UploadExperimentResultAsync(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath))
            {
                throw new ArgumentException("Folder path cannot be null or whitespace.", nameof(folderPath));
            }

            if (!Directory.Exists(folderPath))
            {
                throw new DirectoryNotFoundException($"The specified folder path does not exist: {folderPath}");
            }

            try
            {
                // Initialize the BlobContainerClient with the storage connection string and result container name
                var containerClient = new BlobContainerClient(_config.StorageConnectionString, _config.ResultContainer);
                await containerClient.CreateIfNotExistsAsync(); // Create the container if it doesn't exist

                // List and delete all existing blobs in the container (optional, depending on requirements)
                await foreach (var blobItem in containerClient.GetBlobsAsync())
                {
                    var blobClient = containerClient.GetBlobClient(blobItem.Name);
                    await blobClient.DeleteIfExistsAsync();
                }

                // Iterate through each file in the local folder path
                foreach (var filePath in Directory.GetFiles(folderPath))
                {
                    var fileName = Path.GetFileName(filePath);
                    var blobClient = containerClient.GetBlobClient(fileName);

                    // Upload the file to Azure Blob Storage
                    await blobClient.UploadAsync(filePath, true);

                    // Log successful upload
                    _logger?.LogInformation($"{DateTime.Now} - Uploaded file '{fileName}' to Azure Blob Storage.");
                }
            }
            catch (RequestFailedException ex)
            {
                // Handle Azure SDK request failed exception
                _logger?.LogError($"Failed to upload file to Azure Blob Storage: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
            catch (Exception ex)
            {
                // Handle any other unexpected exceptions
                _logger?.LogError($"An error occurred while uploading files: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
        }


        /// <summary>
        /// Uploads experiment results to Azure Table Storage.
        /// </summary>
        /// <param name="result">The experiment result to be uploaded.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task UploadExperimentResultsTableAsync(List<IExperimentResult> results)
        {
            if (results == null || results.Count == 0)
            {
                throw new ArgumentException("The experiment results list cannot be null or empty.", nameof(results));
            }

            try
            {
                // Create the TableServiceClient and TableClient using the provided connection string and table name from the configuration.
                var tableServiceClient = new TableServiceClient(_config.StorageConnectionString);
                var tableClient = tableServiceClient.GetTableClient(_config.ResultTable);

                // Ensure that the table exists; if it doesn't, create it.
                await tableClient.CreateIfNotExistsAsync();

                foreach (var result in results)
                {
                    if (result == null)
                    {
                        _logger?.LogWarning("Encountered a null experiment result. Skipping.");
                        continue;
                    }

                    try
                    {
                        // Generate a unique RowKey for the new entity.
                        string uniqueRowKey = Guid.NewGuid().ToString();

                        // Ensure that PartitionKey and RowKey are non-null and valid.
                        if (string.IsNullOrWhiteSpace(_config.ResultTable) || string.IsNullOrWhiteSpace(uniqueRowKey))
                        {
                            throw new ArgumentException("PartitionKey or RowKey cannot be null or empty.");
                        }

                        // Create a new TableEntity and populate it with data from the IExperimentResult instance.
                        var entity = new TableEntity(_config.ResultTable, uniqueRowKey)
                        {
                            ["StartTimeUtc"] = result.StartTimeUtc?.ToString("o") ?? string.Empty,
                            ["Timestamp"] = result.Timestamp?.ToString("o") ?? string.Empty,
                            ["EndTimeUtc"] = result.EndTimeUtc?.ToString("o") ?? string.Empty,
                            ["ExperimentId"] = result.ExperimentId ?? string.Empty,
                            ["DurationSec"] = result.DurationSec?.ToString() ?? string.Empty,
                            ["TrainingFileUrl"] = result.TrainingFileUrl ?? string.Empty,
                            ["TestingFileUrl"] = result.TestingFileUrl ?? string.Empty,

                            // Serialize the List<string> to a JSON string to store in Table Storage.
                            ["PredictedLabels"] = result.predictedLabels != null ? JsonSerializer.Serialize(result.predictedLabels) : string.Empty,

                            ["Accuracy"] = result.Accuracy?.ToString() ?? string.Empty,

                            // Include other properties as needed
                            ["OutputFilesProxy"] = result.OutputFilesProxy ?? string.Empty,
                            ["OutputFolderLocation"] = result.OutputFolderLocation ?? string.Empty,
                            ["OutputTableLocation"] = result.OutputTableLocation ?? string.Empty
                        };

                        // Add the entity to the Azure Table.
                        await tableClient.AddEntityAsync(entity);

                        // Log the successful upload of the experiment result.
                        _logger?.LogInformation("Experiment result uploaded successfully: {ExperimentId}", result.ExperimentId);
                    }
                    catch (RequestFailedException ex)
                    {
                        // Handle specific Azure Table Storage errors for each result.
                        _logger?.LogError(ex, "Failed to upload experiment result to Table Storage. Request failed with error: {ErrorCode}. ExperimentId: {ExperimentId}", ex.ErrorCode, result.ExperimentId);
                        Console.Error.WriteLine($"Request failed: {ex.Message}");
                    }
                    catch (Exception ex)
                    {
                        // Handle general exceptions for each result.
                        _logger?.LogError(ex, "An unexpected error occurred while uploading experiment result to Table Storage. ExperimentId: {ExperimentId}", result.ExperimentId);
                        Console.Error.WriteLine($"Unexpected error: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                // Handle exceptions related to table client setup.
                _logger?.LogError(ex, "An unexpected error occurred while initializing Table Storage client or table.");
                Console.Error.WriteLine($"Unexpected error: {ex.Message}");
            }
        }




        /// <summary>
        /// Uploads experiment results as a single file to Azure Blob Storage.
        /// </summary>
        /// <param name="outputTable">The local file path of the results table to upload.</param>
        /// <param name="results">The list of experiment results to include in the upload.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task UploadResultAsync(string outputTable, List<IExperimentResult> results)
        {
            if (string.IsNullOrWhiteSpace(outputTable))
            {
                throw new ArgumentException("Output table file path cannot be null or whitespace.", nameof(outputTable));
            }

            if (results == null || results.Count == 0)
            {
                throw new ArgumentException("Results list cannot be null or empty.", nameof(results));
            }

            try
            {
                // Initialize the BlobContainerClient with the storage connection string and result container name
                var container = new BlobContainerClient(_config.StorageConnectionString, _config.ResultContainer);
                await container.CreateIfNotExistsAsync(); // Create the container if it doesn't exist

                // Get the file name from the outputTable path
                var fileName = Path.GetFileName(outputTable);

                // Get a BlobClient for the specified file name in the container
                var blobClient = container.GetBlobClient(fileName);

                // Upload the file to Azure Blob Storage
                await blobClient.UploadAsync(outputTable, true);

                // Log successful upload
                _logger?.LogInformation($"{DateTime.Now} - Uploaded file '{fileName}' to Azure Blob Storage successfully.");
            }
            catch (RequestFailedException ex)
            {
                // Handle Azure SDK request failed exception
                _logger?.LogError($"Failed to upload file to Azure Blob Storage: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
            catch (Exception ex)
            {
                // Handle any other unexpected exceptions
                _logger?.LogError($"An error occurred while uploading the file: {ex.Message}");
                throw; // Re-throw the exception to ensure it is not silently swallowed
            }
        }
    }
}