using MyCloudProject.Common;
using System;

namespace MyExperiment
{
    /// <summary>
    /// Represents a message request that defines the parameters for running an experiment.
    /// Implements the <see cref="MyCloudProject.Common.IExperimentRequest" /> interface.
    /// </summary>
    internal class ExperimentRequestMessage : IExperimentRequest
    {
        /// <summary>
        /// Gets or sets the unique identifier for the experiment.
        /// </summary>
        public string ExperimentId { get; set; }

        /// <summary>
        /// Gets or sets the training data file associated with the experiment.
        /// </summary>
        public string TrainingDataFile { get; set; }

        /// <summary>
        /// Gets or sets the testing data file associated with the experiment.
        /// </summary>
        public string TestingDataFile { get; set; }

        /// <summary>
        /// Gets or sets the name of the experiment.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the description of the experiment.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Gets or sets the unique identifier of the message.
        /// </summary>
        public string MessageId { get; set; }

        /// <summary>
        /// Gets or sets the receipt identifying the message for processing.
        /// </summary>
        public string MessageReceipt { get; set; }

        /// <summary>
        /// Gets or sets the receipt used to identify and validate message processing.
        /// </summary>
        public string PopReceipt { get; set; }
    }
}