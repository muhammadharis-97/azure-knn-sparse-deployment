using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Defines the contract for all storage operations.
    /// </summary>
    public interface IStorageProvider
    {
        /// <summary>
        /// Receives an experiment request message from an Azure Queue.
        /// </summary>
        /// <param name="token">Cancellation token to stop receiving messages.</param>
        /// <returns>An instance of IExperimentRequest representing the received experiment request, or null if no message is received.</returns>
        public Task<IExperimentRequest> ReceiveExperimentRequestAsync(CancellationToken token);

        /// <summary>
        /// Downloads training and testing files from Azure Blob Storage based on the provided file names and experiment ID.
        /// </summary>
        /// <param name="trainingFileName">The name of the training file to download.</param>
        /// <param name="testingFileName">The name of the testing file to download.</param>
        /// <param name="experimentId">The experiment ID indicating which files to download.</param>
        /// <returns>The local folder path where the files are downloaded.</returns>
        Task<LocalFileWithInputArgs> DownloadInputAsync(string trainingFileName, string testingFileName, string experimentId);


        /// <summary>
        /// Uploads experiment results to Azure Blob Storage.
        /// </summary>
        /// <param name="folderPath">The local folder path containing files to upload.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public Task UploadExperimentResultAsync(string folderPath);

        /// <summary>
        /// Uploads experiment results as a single file to Azure Blob Storage.
        /// </summary>
        /// <param name="outputTable">The local file path of the results table to upload.</param>
        /// <param name="results">The list of experiment results to include in the upload.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>

        public Task UploadResultAsync(string outputTable, List<IExperimentResult> results);

        /// <summary>
        /// Deletes a message from the Azure Queue based on the provided experiment request.
        /// </summary>
        /// <param name="request">The experiment request containing message details.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public Task CommitRequestAsync(IExperimentRequest request);

        /// <summary>
        /// Uploads experiment results to Azure Table Storage.
        /// </summary>
        /// <param name="result">The experiment result to be uploaded.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public Task UploadExperimentResultsTableAsync(List<IExperimentResult> results);
    }
}
