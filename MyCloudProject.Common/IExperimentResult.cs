﻿
using System;
using System.Collections.Generic;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Defines the contract for the result of an experiment.
    /// </summary>
    public interface IExperimentResult
    {
        /// <summary>
        /// Gets or sets the identifier of the experiment associated with this result.
        /// </summary>
        string ExperimentId { get; set; }

        /// <summary>
        /// Gets or sets the URL pointing to the training data file used in the experiment.
        /// </summary>
        string TrainingFileUrl { get; set; }

        /// <summary>
        /// Gets or sets the URL pointing to the testing data file used in the experiment.
        /// </summary>
        string TestingFileUrl { get; set; }

        /// <summary>
        /// Gets or sets the start time of the experiment in UTC.
        /// </summary>
        DateTime? StartTimeUtc { get; set; }

        /// <summary>
        /// Gets or sets the end time of the experiment in UTC.
        /// </summary>
        DateTime? EndTimeUtc { get; set; }

        /// <summary>
        /// Gets or sets the duration of the experiment.
        /// </summary>
        TimeSpan Duration { get; set; }

        public List<string> predictedLabels { get; set; }

        /// <summary>
        /// Gets or sets the accuracy of the experiment results.
        /// </summary>
        double? Accuracy { get; set; }

        /// <summary>
        /// Gets or sets the timestamp associated with the experiment result.
        /// </summary>
        DateTime? Timestamp { get; set; }

        /// <summary>
        /// Gets or sets the elapsed time of the experiment.
        /// </summary>
        TimeSpan? ElapsedTime { get; set; }

        /// <summary>
        /// Gets or sets the duration of the experiment in seconds.
        /// </summary>
        double? DurationSec { get; set; }

        /// <summary>
        /// Gets or sets the proxy to access output files generated by the experiment.
        /// </summary>
        string OutputFilesProxy { get; set; }

        /// <summary>
        /// Gets or sets the location of the output folder containing experiment results.
        /// </summary>
        string OutputFolderLocation { get; set; }

        /// <summary>
        /// Gets or sets the location of the output table file generated by the experiment.
        /// </summary>
        string OutputTableLocation { get; set; }
    }
}