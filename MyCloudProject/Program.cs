using MyCloudProject.Common;
using System;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using MyExperiment;
using Azure.Storage.Queues.Models;
using Azure.Storage.Queues;
using System.Text.Json;
using System.Text;
using System.Collections.Generic;

namespace MyCloudProject
{
    class Program
    {
        /// <summary>
        /// ML 23/24 Investigate and Implement KNN Classifier - Global Variable
        /// </summary>
        private static readonly string _projectName = "ML 23/24 Investigate and Implement KNN Classifier - Global Variable";
        private static IConfigurationRoot _configuration;
        private static ILogger _logger;

        static async Task Main(string[] args)
        {
            var cancellationTokenSource = new CancellationTokenSource();

            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true;
                cancellationTokenSource.Cancel();
            };

            Console.WriteLine($"Started experiment: {_projectName}");

            _configuration = InitHelpers.InitConfiguration(args);
            var configSection = _configuration.GetSection("MyConfig");

            var logFactory = InitHelpers.InitLogging(_configuration);
            _logger = logFactory.CreateLogger("Train.Console");

            _logger.LogInformation($"{DateTime.Now} - Started experiment: {_projectName}");

            IStorageProvider storageProvider = new AzureStorageProvider(configSection, _logger);
            IExperiment experiment = new Experiment(configSection, storageProvider, _logger);

            try
            {
                // Fetch a single experiment request
                IExperimentRequest request = await storageProvider.ReceiveExperimentRequestAsync(cancellationTokenSource.Token);

                if (request == null)
                {
                    _logger.LogWarning("Received a null experiment request.");
                    return;
                }

                try
                {
                    _logger.LogInformation("Processing experiment request: {ExperimentId}", request.ExperimentId);

                    // Download input files
                    var localFileWithInputArgs = await storageProvider.DownloadInputAsync(request.TrainingDataFile, request.TestingDataFile, request.ExperimentId);
                    _logger.LogInformation("Downloaded input files for experiment {ExperimentId}", request.ExperimentId);

                    string trainingDataFolder = localFileWithInputArgs.TrainingDataFolder;
                    string testingDataFolder = localFileWithInputArgs.TestingDataFolder;

                    // Run the experiment
                    List<IExperimentResult> knnResults = await experiment.RunAsync(trainingDataFolder, testingDataFolder);
                    _logger.LogInformation("Experiment {ExperimentId} completed", request.ExperimentId);

                    // Define output locations
                    string outputFolderLocation = Path.Combine(Directory.GetCurrentDirectory(), "OutputFolder");
                    string outputTableLocation = Path.Combine(Directory.GetCurrentDirectory(), "KNN_Results.xlsx");

                    // Upload experiment results to storage
                    await storageProvider.UploadExperimentResultAsync(outputFolderLocation);
                    _logger.LogInformation("Uploaded experiment results for experiment {ExperimentId}", request.ExperimentId);

                    await storageProvider.UploadResultAsync(outputTableLocation, knnResults);
                    _logger.LogInformation("Results uploaded for experiment {ExperimentId}", request.ExperimentId);

                    // Upload results to Table Storage
                    await storageProvider.UploadExperimentResultsTableAsync(knnResults);
                    _logger.LogInformation("Experiment results table updated for experiment {ExperimentId}", request.ExperimentId);

                    // Commit the request
                    await storageProvider.CommitRequestAsync(request);
                    _logger.LogInformation("Experiment request {ExperimentId} committed", request.ExperimentId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "An error occurred while processing the experiment request {ExperimentId}", request.ExperimentId);
                    // Optionally, you can rethrow the exception or handle it as needed
                    throw;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exception occurred while processing experiment request: {Message}", ex.Message);
            }

            _logger.LogInformation($"{DateTime.Now} - Experiment exit: {_projectName}");
        }
    }
}