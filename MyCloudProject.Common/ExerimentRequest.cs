using System;
using System.Collections.Generic;
using System.Text;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Defines the contract for the message request that will run an experiment.
    /// </summary>
    public interface IExperimentRequest
    {
        /// <summary>
        /// Gets or sets the unique identifier for the experiment.
        /// </summary>
        string ExperimentId { get; set; }

        /// <summary>
        /// Gets or sets the path or identifier for the training file.
        /// </summary>
        string TrainingDataFile { get; set; }

        /// <summary>
        /// Gets or sets the path or identifier for the testing file.
        /// </summary>
        string TestingDataFile { get; set; }

        /// <summary>
        /// Gets or sets the name of the experiment.
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// Gets or sets a brief description of the experiment.
        /// </summary>
        string Description { get; set; }

        /// <summary>
        /// Gets or sets the unique identifier for the message.
        /// </summary>
        string MessageId { get; set; }

        /// <summary>
        /// Gets or sets the receipt handle for the message.
        /// </summary>
        string MessageReceipt { get; set; }

        /// <summary>
        /// Gets or sets the receipt handle for the message.
        /// </summary>
        string PopReceipt { get; set; }
    }
}
