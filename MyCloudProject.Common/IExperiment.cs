using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Defines the interface for an experiment, providing methods for running the experiment with specified data.
    /// </summary>
    public interface IExperiment
    {
        /// <summary>
        /// Runs the experiment using the provided training and testing data folders.
        /// </summary>
        /// <param name="trainingDataFolder">The local folder path containing the training data files.</param>
        /// <param name="testingDataFolder">The local folder path containing the testing data files.</param>
        /// <returns>A task representing the asynchronous operation, containing the result of the experiment.</returns>
        Task<List<IExperimentResult>> RunAsync(string trainingDataFolder, string testingDataFolder);
    }
}
