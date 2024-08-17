using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using Microsoft.Extensions.Logging.Console;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.IO;

namespace MyCloudProject.Common
{
    /// <summary>
    /// Provides helper methods for initializing logging and configuration in the cloud project.
    /// </summary>
    public static class InitHelpers
    {
        /// <summary>
        /// Initializes the logging infrastructure for the cloud project.
        /// </summary>
        /// <param name="configRoot">The root configuration object containing logging settings.</param>
        /// <returns>An <see cref="ILoggerFactory"/> instance configured according to the provided settings.</returns>
        public static ILoggerFactory InitLogging(IConfigurationRoot configRoot)
        {
            if (configRoot == null)
            {
                throw new ArgumentNullException(nameof(configRoot), "Configuration root cannot be null.");
            }

            return LoggerFactory.Create(logBuilder =>
            {
                // Add logging configuration from appsettings or environment variables
                logBuilder.AddConfiguration(configRoot.GetSection("Logging"));

                // Configure console formatte with options
                logBuilder.AddConsole(ConsoleFormatteroptions =>
                {
                    ConsoleFormatteroptions.IncludeScopes = true; // Include scope information in console output
                }).AddDebug(); // Enable debug logging
            });
        }

        /// <summary>
        /// Initializes configuration for the cloud project based on appsettings files and environment variables.
        /// </summary>
        /// <param name="args">Command-line arguments to override configuration settings.</param>
        /// <returns>An <see cref="IConfigurationRoot"/> instance representing the loaded configuration.</returns>
        public static IConfigurationRoot InitConfiguration(string[] args)
        {
            // Determine the environment name based on environment variable
            var environmentName = Environment.GetEnvironmentVariable("MYCLOUDPROJECT_ENVIRONMENT");

            var builder = new ConfigurationBuilder()
                .SetBasePath(AppContext.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);

            // Load environment-specific configuration if specified
            if (!string.IsNullOrEmpty(environmentName))
            {
                var envFileName = $"appsettings.{environmentName}.json";
                if (File.Exists(Path.Combine(AppContext.BaseDirectory, envFileName)))
                {
                    builder.AddJsonFile(envFileName, optional: false, reloadOnChange: true);
                }
            }

            // Add command-line arguments to override configuration settings
            if (args != null && args.Length > 0)
            {
                builder.AddCommandLine(args);
            }

            // Add environment variables to configuration
            builder.AddEnvironmentVariables();

            try
            {
                // Build and return the configuration root
                return builder.Build();
            }
            catch (Exception ex)
            {
                // Handle potential configuration loading issues
                throw new InvalidOperationException("Failed to build configuration.", ex);
            }
        }
    }
}
