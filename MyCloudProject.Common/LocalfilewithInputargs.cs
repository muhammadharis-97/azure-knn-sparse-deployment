using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Represents the local file paths for training and testing data along with any associated input arguments.
    /// </summary>
    public class LocalFileWithInputArgs
    {
        /// <summary>
        /// Gets or sets the folder path where the training data is stored.
        /// </summary>
        public string TrainingDataFolder { get; set; }

        /// <summary>
        /// Gets or sets the folder path where the testing data is stored.
        /// </summary>
        public string TestingDataFolder { get; set; }
    }
}


