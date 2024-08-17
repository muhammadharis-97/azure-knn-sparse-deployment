# Implementation of KNN Classifier for Sequence Prediction - Azure Cloud Implementation


## Key Features

- [x] **Microsoft Azure Integration:**  
Deployed on Azure for robust cloud infrastructure, ensuring high availability and scalability for our SE project.

- [x] **Docker for Containerization:** 
Utilizes Docker to package applications, providing consistent environments and simplifying deployment.

- [x] **KNN Classifier:**   
Supports large datasets (200 JSON Files[Multiple Files]) with more than (2000+ Sequences) for efficient and accurate data classification.

- [x] **Monitoring and Logging:**  
  Real-time insights via Azure Monitor and Log Analytics for performance tracking and issue resolution.




## Table of Contents

1. [Introduction](#introduction)
2. [Essentials - SE Project](#essentials---se-project)
3. [Project Architecture](#project-architecture)
4. [Implemented Technology](#implemented-technology)
5. [Source Files - CC Project](#source-files---cc-project)
6. [Project Implementation](#project-implementation)
7. [Azure Deployment](#azure-deployment)
8. [How to Run the Project](#how-to-run-the-project)
9. [Output Table](#output-table)
10. [Future Scope](#future-scope)
11. [Conclusion](#conclusion)
12. [References](#references)



## Introduction:
This section presents the overview of implementing the k-nearest neighbor (KNN) machine learning algorithm for predicting outcome variables based on input variables. Leveraging the capabilities of Hierarchical Temporal Memory (HTM) for learning complex temporal patterns, our study focuses on predicting sequences of random sequences between 0-48 floating numbers. We integrate the KNN model with the Neocortex API to efficiently classify sequences, now deployed on the Azure Cloud.

The KNN model receives input data as Sparse Distributed Representations (SDR) from HTM stored in the Container Blob Storage. We construct a dataset comprising multiple sequence SDRs, each with varying values within a defined threshold. To segregate data, we have two storage containers named trainingdata and testingdata, and one container for output. All of these details are discussed below.



## Essentials - SE Project
1.  [Paper](https://github.com/zakaahmadchishti/Global_Variables/blob/ZAKA-AHMED/My%20Project%3AKNN%20/Report-Final/Investigate%20and%20Implement%20KNN%20Classifier_Global%20Variable-paper.pdf)
2.  [Readme File](https://github.com/zakaahmadchishti/Global_Variables/blob/Muhammadharis/Myproject/Documentation/README.md)
3.  [Issues](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/issues?q=is%3Aissue+Global+Variable)
4.  [Project Source Files](https://github.com/zakaahmadchishti/Global_Variables/tree/9174f2e6f5ab8df3f299c91e92dcba7cd6e5edef/Myproject/KNN)


## Project Architecture

This section explains the project architecture. The below diagram represents the components, the main structure, and the communication functionality. The diagram also shows the critical modules that we have used in the project and also tries to represent their functionality with other modules. 

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/192119616790c79064822c1091240adcb8ac520e/Source/MyCloudProjectSample/Documentation/Screenshots/Project%20Architecture.png" width="800" alt="TEAM Global Variable: Project Architecture" />
</p>
<p align="center">
  <em>Figure 1: Project Architecture </em>
</p>

Below is the detail of Elements which are used through-out the project. 

| Element                           | Description                                                                   |
|-----------------------------------|-------------------------------------------------------------------------------|
| **SE Source Files**               | Source code files written by the developer (`KNNClassifier.cs`, `MultisequenceLeaning.cs`, `SquenceLeaning.cs`). |
| **Visual Studio**                 | Integrated Development Environment used for code development and debugging.               |
| **Docker Container**              | A virtualized environment used to package the application and its dependencies.|
| **GitHub**                        | Version control system where the code is stored and managed.                  |
| **Global Variable (TEAM)**        | Collaborative team contributing to the codebase.                              |
| **Image File**                    | File created from the Docker container, containing the application package.   |
| **Azure Container Registry**      | Service for storing and managing container images.                            |
| **Azure Container Instance**      | Service for running Docker containers on Azure.                               |
| **Logs and Monitoring**           | Tools used for tracking the application's performance and diagnosing issues.  |
| **Azure Cloud Environment**       | Cloud platform where the application is deployed and run.                     |
| **Storage Account**               | Service for storing various types of data.                                    |
| **Queues**                        | Service for managing asynchronous messaging between different components.     |
| **Blob Containers**               | Service for storing large amounts of unstructured data.                       |
| **Tables**                        | Service for storing structured, non-relational data.                          |
| **Trigger Input**                 | Event or action that initiates a process.                                     |
| **Multiple Input Files**          | Various files are needed as input for the application.                            |
| **Output File**                   | Resultant file generated by the application.                                  |

More discussion about the Input and Output file is given below in the Project Implementation Section. 

#### KNN Architecture - SE Project 
The KNN implementation is divided into three main components. The first and primary component is the KNNClassifier class, which includes the key methods: distance, vote, and classify. The second component is the IndexAndDistance class, responsible for managing the indices and distances of data points. The third component, also part of the KNNClassifier, includes methods to handle classification tasks.

	•	The distance method calculates the distance between a new, unlabeled data point and the existing data points.
	•	The vote method creates a voting table based on the calculated distance matrix, determining the most likely class for the new data point.
	•	The classify method uses the results from the vote method to assign a class label to the unknown data point.

It’s important to note that we are using JSON files as our dataset, with multiple file inputs as discussed in the CC Project’s Project Implementation Section.

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/ead459b9083b04c094ceced1e1e5ea11fba9e73d/Source/MyCloudProjectSample/Documentation/Screenshots/SE-Project.png" width="800" alt="TEAM Global Variable: KNN Architecture" />
</p>
<p align="center">
  <em>Figure 2: KNN Impementation (SE Project) </em>
</p>

The diagram illustrates the processes of training and testing a KNN classifier using random sequences. In the training phase, a set of 200 random sequences is created. This will be our starting point. These sequences undergo encoded sequences and spatial pooling to create Sparse Distributed Representations (SDRs), which form the features and labels of the training dataset. The KNN classifier is trained on this dataset by calculating distances, selecting nearest neighbors, and using a voting mechanism to determine class labels. In the testing phase, a separate set of random sequences is processed (Which we selected from our 200 random sequences) similarly to generate testing data SDRs. These features are classified by the trained KNN model, which predicts the class labels based on the learned patterns. The accuracy of these predictions is then evaluated, with higher accuracy expected if most of the training labels are consistent with the testing labels.



## Implemented Technology
Here is the list of technologies, we have used in this project.

| Application | Platform |
|---|---|
| Programming Language | C# |
| Containerization | Docker |
| Cloud Platform | Azure |
| Input | Blob Container Storage |
| Output | Table Storage |
| Triggering | Queue |



## Source Files - CC Project 
1.  [Readme File](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/90a51356d8de5e6d2960f71b4685265605866a82/Source/MyCloudProjectSample/Documentation/Implementation%20of%20KNN%20Classifier%20for%20Sequence%20Prediction.md)
2.  [Issues](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/issues?q=is%3Aissue+Global+Variable)
3.  [Project Source Files](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/tree/Global-Variable/Source/MyCloudProjectSample)
4.  [Training Dataset](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/tree/8d539862a2339650f3ede381964211e3956d3fcb/Myproject/KNN/dataset/trainingdatafolder)
5.  [Testing Dataset](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/tree/Global-Variable/Myproject/KNN/dataset/testingdatafolder)


## Project Implementation
Before proceeding, it’s important to highlight the changes made in this project compared to the previous SE Project. The key update is in the input file structure for training and testing. Initially, we used a single input file for all the data. However, due to the need for a larger dataset, we have now transitioned to using multiple input files in the form of JSON files. Previously, the SE Project utilized only three types of sequence classes as input data. In contrast, the current project has expanded to 200 sequence classes, with each sequence stored in a separate JSON file. This means we now have 200 JSON files, with each sequence class containing 10 attributes. An example of one of these JSON files is provided in the Input File Section.

The Important components are the input files and the queue which are discussed in the below sections. 



### Input Files (Training Data)
We are using 200 JSON Files in the Project. Which are used as the training dataset. These 200 Files(JSON) means we have 200 sequences and every sequence has 10 attributes(SDRs). Below is the one JSON File sample. 


```json
{"SequenceName":"S1","SequenceData":[4898,5102,5461,6476,6640,6702,6884,6923,7034,7072,7077,7128,7170,7226,7598,8002,8177,8431,8504,8559]}
{"SequenceName":"S1","SequenceData":[4898,5102,5461,6476,6640,6702,6884,6923,7034,7072,7077,7128,7170,7226,7598,8002,8177,8431,8504,8559]}
{"SequenceName":"S1","SequenceData":[5344,5399,5417,5529,6406,6464,6775,6848,7021,7034,7117,7234,7351,7411,7511,7534,7608,7686,7913,8128]}
{"SequenceName":"S1","SequenceData":[5344,5399,5417,5529,6406,6464,6775,6848,7021,7034,7117,7234,7351,7411,7511,7534,7608,7686,7913,8128]}
{"SequenceName":"S1","SequenceData":[4615,5599,5944,6610,6657,6691,6991,7170,7187,7257,7319,7331,7389,7508,7535,7590,7618,7698,7960,8185]}
{"SequenceName":"S1","SequenceData":[4615,5599,5944,6610,6657,6691,6991,7170,7187,7257,7319,7331,7389,7508,7535,7590,7618,7698,7960,8185]}
{"SequenceName":"S1","SequenceData":[5564,5604,6292,6321,6517,6525,6630,6879,7036,7056,7173,7218,7239,7256,7394,7439,7542,7556,7671,7973]}
{"SequenceName":"S1","SequenceData":[5564,5604,6292,6321,6517,6525,6630,6879,7036,7056,7173,7218,7239,7256,7394,7439,7542,7556,7671,7973]}
{"SequenceName":"S1","SequenceData":[5457,6462,6718,6773,6780,6860,6941,7012,7068,7077,7136,7233,7309,7331,7526,7613,7644,7658,7912,8433]}
{"SequenceName":"S1","SequenceData":[5457,6462,6718,6773,6780,6860,6941,7012,7068,7077,7136,7233,7309,7331,7526,7613,7644,7658,7912,8433]}

```
The above sequence represents the S1 Sequence-Type and has 10 attributes(SDRs). Next, the below, code below shows the representation of reading files and then separating the label and SDRs. Which are used further to calculate K-Nearest. 

```csharp
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
            Classifierleaning.Featursandlabel(sequenceDataEntriestraining, out List<List<double>> trainingFeatures, out List<string> trainingLabels);
            // Extract the testing data labels and features.
            Classifierleaning.Featursandlabel(sequenceDataEntriestesting, out List<List<double>> testingFeatures, out List<string> testingLabels);
```

Below is the reference to the training data used in the project. 

```csharp
InputFiles = "https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/tree/8d539862a2339650f3ede381964211e3956d3fcb/Myproject/KNN/dataset/trainingdatafolder",
 //See project Training Dataset for more information 
```



<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/b83d8382e638317d41510653557ed047fedece70/Source/MyCloudProjectSample/Documentation/Screenshots/Input%20files.png.png" width="800" alt="TEAM Global Variable: Training Dataset" />
</p>
<p align="center">
  <em>Figure 3: Training Dataset(Input)</em>
</p>







### Input (Testing Data)  
The testing data are random sequences which we will test and determine in which class they will fall. As below we have shown one of the sample formats of testing data. Against every sequence, votes are calculated and determined on a voting basis which is the nearest sequence from the training data. 


```json
{"SequenceName":"S152","SequenceData":[12961,14365,14418,14447,14468,14517,14747,14774,14788,14957,14998,15001,15029,15083,15132,15443,15484,15532,15584,15772]}
{"SequenceName":"S178","SequenceData":[20094,21326,21524,21576,21668,21834,21916,21925,21971,22007,22268,22278,22307,22333,22390,22564,22752,22851,22958,23795]}
{"SequenceName":"S180","SequenceData":[9274,9332,9401,9957,10035,10069,10089,10348,10379,10442,10642,10675,10724,10809,10865,11114,11334,11806,12020,12124]}
{"SequenceName":"S153","SequenceData":[14331,14368,14405,14464,14512,14963,14980,15085,15136,15180,15263,15331,15371,15429,15479,15534,15552,15585,15717,15774]}
{"SequenceName":"S179","SequenceData":[5157,5646,5767,6074,6699,6862,6884,7133,7157,7209,7296,7371,7534,7749,7873,7933,7995,8094,8110,8479]}
```

As these SDRs are random, KNN finds the one with high votes. Select the closest to the specific class. We have created multiple test files similar to those shown above which are used as our testing data. 



```csharp
InputFiles = "https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/tree/8d539862a2339650f3ede381964211e3956d3fcb/Myproject/KNN/dataset/testingdatafolder",
 //See project Training Dataset for more information 
```

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/8d539862a2339650f3ede381964211e3956d3fcb/Source/MyCloudProjectSample/Documentation/Screenshots/Testing%20Files.png" width="800" alt="TEAM Global Variable: Testing Dataset" />
</p>
<p align="center">
  <em>Figure 4: Testing Dataset (Input) </em>
</p>



### Triggering Input (Queue)
The main purpose behind using the Queue is to store and retrieve messages. This queue likely triggers processing or workflows related to your KNN algorithm or other tasks. Below is the Message Format of our Queue. We have created two cases. The first case considered specific test files and in the second case all of the testing files were used. Examples are shown below. 

Case 1: Using ALL Testing Files:
In the First case, we have to give the specific file number. So that only that specific file will be used for testing. 

```json
{
  "ExperimentId": "1",
  "Name": "KNN Classifier Prediction",
  "Description": "KNN Classifier",
  "TrainingDataFile": "ALL",
  "TestingDataFile": "testfile1.json"
}
```

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/8d539862a2339650f3ede381964211e3956d3fcb/Source/MyCloudProjectSample/Documentation/Screenshots/Queue.png" width="800" alt="TEAM Global Variable: Testing Dataset" />
</p>
<p align="center">
  <em>Figure 5: Queue (Triggering Input) (</em>
</p>


Case 2: Using Single Testing File.
Below is the JSON format when we are using the ALL Files as the testing dataset. 

```json
{
  "ExperimentId": "2",
  "Name": "KNN Classifier Prediction",
  "Description": "KNN Classifier",
  "TrainingDataFile": "ALL",
  "TestingDataFile": "ALL"
}
```


ExperimentRequestMessage Method which is defined in ExerimentRequestMessage.cs. 

```csharp
 public ExperimentRequestMessage(
   string experimentId,
   string trainingDataFile,
   string testingDataFile,
   string name,
   string description,
   string messageId,
   string messageReceipt)
    {
     ExperimentId = experimentId ?? throw new ArgumentNullException(nameof(experimentId));
     TrainingDataFile = trainingDataFile ?? throw new ArgumentNullException(nameof(trainingDataFile));
     TestingDataFile = testingDataFile ?? throw new ArgumentNullException(nameof(testingDataFile));
     Name = name ?? throw new ArgumentNullException(nameof(name));
     Description = description;
     MessageId = messageId ?? throw new ArgumentNullException(nameof(messageId));
     MessageReceipt = messageReceipt ?? throw new ArgumentNullException(nameof(messageReceipt));
        }
    }
```


### Algorithm Background and Operations
The K-Nearest Neighbors (KNN) algorithm is a non-parametric, instance-based learning method primarily used for classification and regression. It operates on the principle of similarity, where the output is determined by the ‘k’ most similar instances (neighbors) in the dataset.

In the SE project, the input data is represented as Sparse Distributed Representations (SDRs) generated by the Hierarchical Temporal Memory (HTM) model. The algorithm calculates the distance between data points using various metrics like Euclidean, Manhattan, Minkowski, or Hamming distance, which influences the accuracy and performance of the model. Well, we only used Euclidean. For classification tasks, KNN assigns a class to a data point based on the majority vote among its ‘k’ nearest neighbors. For regression tasks, the output is the average of the values of ‘k’ nearest neighbors. The value of ‘k’ is crucial; a smaller ‘k’ can lead to overfitting, while a larger ‘k’ can result in underfitting. Cross-validation is often used to find the optimal ‘k’.

The project integrates KNN with the HTM model, where HTM generates SDRs that serve as input to the KNN classifier. The dataset consists of multiple sequence SDRs of a random sequences, split into training data and testing data. The KNN model processes sequence SDRs and achieves high accuracy rates. The primary use case in this project is the classification of random sequences (in 200 Classes). The integration with the Neocortex API enhances the model’s efficiency in classifying sequences.

KNN is simple and effective for various classification tasks, robust with respect to different data types and distributions. However, it is computationally intensive for large datasets, sensitive to the choice of ‘k’ and distance metrics, and can struggle with imbalanced datasets. The KNN algorithm, when combined with HTM and SDRs, demonstrates high accuracy in sequence classification tasks, showcasing the potential for improved predictive performance in various machine learning applications.

As already discussed, the main difference is dataset is changed in the CC Project. We have extended our dataset. 




## Azure Deployment

### Azure Configuration Details
All the necessary configurations in Azure have been meticulously set up, which include the creation of various resources essential for our project. Below are the detailed configurations:

| Configuration               | Details                             |
| --------------------------- | ----------------------------------- |
| **Resource Group**          | GlobalVariable                      |
| **Azure Container Registry**|  globalvariablecc                       |
| **Azure Container Instances**|  globalvariableci                    |
| **Azure Storage Account**           | ccprojectstoragek                   |
| **Container 1**             | trainingdatafolder          |
| **Container 2**             | testingdatafolder           |
| **Container 3**             | knnresults                  |
| **Queue**                   | knntrigger-queue            |
| **Table**                   | resultsknn                  |

The above shows all the detailed names we used during cloud configuration. The connection string is also given below. 

#### Detailed Breakdown

| **Configuration**             | **Details**                             |
|-------------------------------|-----------------------------------------|
| **Resource Group**            | **Name:** GlobalVariable<br>**Purpose:** A logical container for grouping related Azure resources. All resources related to your project are contained within this resource group for ease of management and organization. |
| **Azure Container Registry**  | **Name:** globalvariablecc<br>**Purpose:** Azure Container Registry (ACR) is used to store and manage Docker container images and artifacts. It allows you to build, store, and manage container images for deployment to Azure Kubernetes Service (AKS), Azure Container Instances (ACI), or other container-based services.<br>**Action Required:** Specify the name of the Azure Container Registry, e.g., `mycontainerregistry`. |
| **Azure Container Instances** | **Name:** globalvariableci<br>**Purpose:** Azure Container Instances (ACI) allows you to run containers in the cloud without managing virtual machines or needing container orchestration. It’s suitable for running lightweight, stateless containers.<br>**Action Required:** Specify the name and details of your Azure Container Instances, e.g., `mycontainerinstance`. |
| **Azure Storage Account**     | **Name:** ccprojectstoragek<br>**Purpose:** Provides scalable and secure cloud storage for your data. It supports various types of storage, including blobs, files, queues, and tables. |
| **Containers in Storage Account** | **Container 1:**<br>**Name:** trainingdatafolder<br>**Purpose:** Used to store training data for your machine learning or data processing tasks.<br>**Container 2:**<br>**Name:** testingdatafolder<br>**Purpose:** Used to store testing data, which is used to validate and test your models or data processing workflows.<br>**Container 3:**<br>**Name:** knnresults<br>**Purpose:** Stores results from K-Nearest Neighbors (KNN) algorithm runs or any other related output. |
| **Queue**                     | **Name:** knntrigger-queue<br>**Purpose:** Used to store and retrieve messages. This queue likely triggers processing or workflows related to your KNN algorithm or other tasks. |
| **Table**                     | **Name:** resultsknn<br>**Purpose:** Azure Table Storage is used for storing structured NoSQL data. It’s often used for storing large amounts of data that doesn’t need complex queries or relational features.<br>**Action Required:** The name of the Azure Table Columbs, e.g., `PredictedLabels`, 'Accuracy', 'resultsknn'. |


Once these configurations are finalized, you can view and manage all details within the designated Resource Group in the Azure portal. Which will provide a consolidated view for efficient monitoring, management, and troubleshooting.



Details are mentioned in appsettings.json shown below:

```json
  "GroupId": "Global Variable",
      "StorageConnectionString": "DefaultEndpointsProtocol=https;AccountName=ccprojectstoragek;AccountKey=gZuELYCXtktGoeZqP8EZo+4C1izqAA4ALkX17yMmhBXeeUBmMVrDdH2/A542zbrLgukNer0ul/8V+AStpUpJDw==;EndpointSuffix=core.windows.net",
      "TrainingContainer": "trainingdatafolder",
    "TestingContainer": "testingdatafolder",
    "ResultContainer": "knnresults",
    "ResultTable": "resultsknn",
    "PredictionlabelTable": "predictedLabels",
    "AccuracyTable": "accuracyknn",
    "Queue": "knntrigger-queue"
```

By setting up these configurations, we have established a robust foundation for deploying, managing, and scaling our KNN classifier application on the Azure cloud platform.



<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/96a4ad67c08df5424532d33686f9cfa3f4e2171f/Source/MyCloudProjectSample/Documentation/Screenshots/Container%20Registry.png" width="800" alt="TEAM Global Variable: Connection String" />
</p>
<p align="center">
  <em>Figure 6: Container Registry </em>
</p>



<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/15c6a4d148163d78a67cabec8d058ada02d53d6a/Source/MyCloudProjectSample/Documentation/Screenshots/Instance%20Created.png" width="800" alt="TEAM Global Variable: Connection String" />
</p>
<p align="center">
  <em>Figure 7: Container Instance </em>
</p>

The above picture shows our connection string which is used to establish a connection between azure and the physical machine. 

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/96a4ad67c08df5424532d33686f9cfa3f4e2171f/Source/MyCloudProjectSample/Documentation/Screenshots/Access%20Key%20Registry.png" width="800" alt="TEAM Global Variable: Connection String" />
</p>
<p align="center">
  <em>Figure 8: Key </em>
</p>

## How to Run the Project

This section provides a detailed approach to executing the experiment. Below is the 




### Step 1: Start the Container Instance
As a first step please navigate yourself to start the Instance. 


<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/9ac4b94af30f0393c334799d0c7bce0fc4e1f813/Source/MyCloudProjectSample/Documentation/Screenshots/Instance%20Created.png" width="800" alt="TEAM Global Variable: Connection String" />
</p>
<p align="center">
  <em>Figure 9: New Instance Created </em>
</p>



### Step 3: Add Input Data 
Next, Step is adding the Input Data File (Github) for Testing as well as for training into the Blob Container. 



##### 1. Adding a Training Data || trainingdatafolder
Here you can see that we have to add the Training Data in the Container. So, the Input Training Files are added Now.

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/b83d8382e638317d41510653557ed047fedece70/Source/MyCloudProjectSample/Documentation/Screenshots/Input%20files.png.png" width="800" alt="TEAM Global Variable: Training Dataset" />
</p>
<p align="center">
  <em>Figure 10: Adding a Training Dataset </em>
</p>


You can also Upload your files by keeping the formatting in mind. 

##### 2. Adding a Testing Data || testingdatafolder
Here you can see that we have to add the Testing Data in the Container. 

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/8d539862a2339650f3ede381964211e3956d3fcb/Source/MyCloudProjectSample/Documentation/Screenshots/Testing%20Files.png" width="800" alt="TEAM Global Variable: Testing Dataset" />
</p>
<p align="center">
  <em>Figure 11: Testing Dataset </em>
</p>

Now, the next step will be adding the Queue Message. 


### Step 3: Queue message to the Experiment || knntrigger-queue 
Now we have to add the Queue Message. As we talked about earlier, we have two cases for giving the Input Queue. Let's say consider case 2. We have to add the message to the Queue. As discussed in Triggering Input (Queue) section above.


<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/8d539862a2339650f3ede381964211e3956d3fcb/Source/MyCloudProjectSample/Documentation/Screenshots/New%20Adding.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 12: Queue Message </em>
</p>

Here the Instance is waiting for the triggering Queue. As soon as we enter the triggering message into the queue, the instance experiment execution gets started. 

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/96a4ad67c08df5424532d33686f9cfa3f4e2171f/Source/MyCloudProjectSample/Documentation/Screenshots/Waiting%20for%20Queue.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 13: Adding a Queue Message </em>
</p>




### Step 4: Monitoring of the Progress || globalvariableci
We can review the container instance logs to monitor the progress of the experiment. Be sure to document any significant events or anomalies you observe, as this information will be important for future reference.

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/96a4ad67c08df5424532d33686f9cfa3f4e2171f/Source/MyCloudProjectSample/Documentation/Screenshots/Accuracy%20with%20the%20test%20Files.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 14: Output Accuracy of KNN </em>
</p>



<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/96a4ad67c08df5424532d33686f9cfa3f4e2171f/Source/MyCloudProjectSample/Documentation/Screenshots/Accuracy%20instance.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 15: Output Accuracy of KNN </em>
</p>



### Step 5: Output || Result container & outputtable
As when the execution reaches its end, we will get all the outputs which include the Result Container (knnresults) and the Table (resultsknn). As we have shown below the Snapshots of the Output Container and the Table for reference. 


<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/54ad933713f57bcb56df5e21627661e7044f7fad/Source/MyCloudProjectSample/Documentation/Screenshots/Output%20Container.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 14: Output Container </em>
</p>

The below screenshot shows the Output table as the table consists of multiple columns explained in the below table (Section Output Table).

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/19d3a06af0d273dadeffb8a51dd33af0bb2dec1d/Source/MyCloudProjectSample/Documentation/Screenshots/Ouput%20Table.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 15: Output Table </em>
</p>




## Output Table
As we get an Output on the Result Container (knnresults) and the Table (resultknn). The text file can be look like this. 

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/54ad933713f57bcb56df5e21627661e7044f7fad/Source/MyCloudProjectSample/Documentation/Screenshots/Output%20Txt%20file.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 16: Output Text Files </em>
</p>



The Table consists of the Accuracyknn and PredicitedLables. Which are already defined by their names.

<p align="center">
  <img src="https://github.com/UniversityOfAppliedSciencesFrankfurt/se-cloud-2023-2024/blob/Global-Variable/Source/MyCloudProjectSample/Documentation/Screenshots/Output.png" width="800" alt="TEAM Global Variable: Adding a Queue Message" />
</p>
<p align="center">
  <em>Figure 17: Output </em>
</p>

We have also created an Excel (KNN_Results.xlsx) file which has the following columns list. 

This is what the Table looks like. Following are the column's details. 

| **Column Name**              | **Heading**                          | **Explanation**                                                                 |
| --------------------------- | -----------------------------------  | ------------------------------------------------------------------------------  |
| **Column1**                  | Timestamp                            | The exact date and time when the experiment or event started.                    |
| **Column2**                  | EndTimeUtc                           | The exact date and time when the experiment or event ended, in UTC format.       |
| **Column3**                  | ExperimentId                         | A unique identifier assigned to each experiment for tracking and reference.      |
| **Column4**                  | DurationSec                          | The total duration of the experiment in seconds, calculated as the difference between `EndTimeUtc` and `Timestamp`. |
| **Column5**                  | TrainingFileUrl                     | A reference or link to the training files generated during the experiment, often pointing to a storage location. |
| **Column6**                  | TestingFileUrl                       | The URL or path to the file used for testing the model or experiment results.     |
| **Column7**                  | predictedLabels                      | The labels predicted by the model during the testing phase, often used for evaluating performance. |
| **Column8**                  | Accuracy                             | The accuracy of the model's predictions, represented as a percentage or decimal value. |
| **Column9**                  | OutputFilesProxy                     | The location where output files get updated. |


All the required details regarding the output are discussed above. Now we move towards future scope and conclusion. 


 
## Future Scope
One area of focus could be scalability and optimization, which includes performance tuning to handle larger datasets and reduce computation time, as well as implementing parallel processing techniques to speed up the KNN classification. Additionally, integrating the KNN classifier with other machine learning models, such as combining it with neural networks or decision trees to create hybrid models, can offer better accuracy and robustness. Incorporating Automated Machine Learning (AutoML) tools to automatically select the best model and hyperparameters for different datasets is another avenue for future development. Furthermore, exploring advanced distance metrics could improve the classifier’s performance by finding more appropriate measures for specific types of data. Expanding the range of applications by adapting the KNN classifier for different domains, such as image recognition, natural language processing, and real-time analytics, can also broaden the project’s impact. Finally, enhancing the user interface and user experience for managing and monitoring the classification process in the cloud will make the system more user-friendly and accessible to a broader audience.



## Conclusion
The implementation of the KNN Classifier for random sequence prediction is a significant step toward predicting sequence classes with high accuracy. This project showcases the successful integration of the KNN algorithm into a cloud environment using Azure services, future improvements can provide us with a scalable, reliable, and efficient solution for large-scale sequence prediction. 



## References
1. [SE-Project-Paper](https://github.com/zakaahmadchishti/Global_Variables/blob/ZAKA-AHMED/My%20Project%3AKNN%20/Report-Final/Investigate%20and%20Implement%20KNN%20Classifier_Global%20Variable-paper.pdf)
2. [Neocortex API](https://github.com/ddobric/neocortexapi)
3. [KNN concepts](https://www.ibm.com/topics/knn)
4. [Microsoft Azure Deployment Guide](https://learn.microsoft.com/en-us/azure/app-service/deploy-best-practices)









