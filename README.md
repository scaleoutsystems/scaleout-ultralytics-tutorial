# Scaleout Edge Ultralytics Tutorial

## Introduction
This tutorial will guide you through the process of implementing a federated learning setup using the **Scaleout Edge** platform in combination with **Ultralytics** YOLOv8 models. Federated learning allows multiple clients to collaboratively train a global model without sharing their local datasets, ensuring data privacy and security. This tutorial is designed for users familiar with machine learning and federated learning concepts and will provide step-by-step instructions on configuring and running a complete federated learning workflow.

By the end of this tutorial, you will have built a distributed training environment where clients independently train local models, and a global model is aggregated on the server. You will also learn how to use Ultralytics YOLOv8 models for object detection tasks in this federated setting.

1.	Starting the server in Scaleout Edge – Learn how to initiate the central server that coordinates federated learning activities.
2.	Cloning the repository – Set up the project by cloning the necessary repository for configuration and deployment.
3.	Installing prerequisites – Install all required dependencies for the client environments.
4.	Setting up the dataset – Properly structure and configure your dataset to be used with Ultralytics models.
5.	Setting up configurations – Configure the model, dataset paths, and training parameters in `client_config.yaml`.
6.	Building the compute package – Build the compute package to prepare for the training process.
7.	Initializing the seed model – Generate the initial model to start the training process.
8.	Initializing the server-side – Set up the server-side of the federated learning system.
9.	Connecting and starting the clients – Connect the clients that will participate in the federated training.
10.	Training the global model – Observe how the global model improves through the aggregation of client updates and monitor training progress.

By following these steps, you will not only gain hands-on experience with the Scaleout Edge platform but also learn how to integrate object detection tasks with YOLOv8 in a federated learning environment.

## Prerequisites
-  `Python ==3.12` <https://www.python.org/downloads>
-  `uv` <https://docs.astral.sh/uv/getting-started/installation/>

## Step 1: Starting the server in Scaleout Edge
Request a hosted server on Scaleout Edge <https://www.scaleoutsystems.com/>

Once you are logged in on your Scaleout Edge domain, you need to start a new project by clicking on the `New project` button.
This initializes the server which later will be used to run the federated learning process.

## Step 2: Cloning the repository
Next, you need to clone the repository:
```bash
git clone https://github.com/scaleoutsystems/fedn-ultralytics-tutorial
```
Then navigate into the repository:
```bash
cd fedn-ultralytics-tutorial
```
This repository contains all the necessary files and configurations for the federated learning setup.

## Step 3: Installing prerequisites
Install all dependencies using UV:
```bash
uv sync
```
This will create a virtual environment and install all required packages as defined in `pyproject.toml`.

## Step 4: Setting up the dataset
Organize your data into a folder named `fed_dataset` inside a `datasets` directory at the root of the repository. Your folder structure should look like this:
```
datasets/
  fed_dataset/
    train/
      images/
        image1.jpg
        image2.jpg
        ...
      labels/
        image1.txt
        image2.txt
        ...
    valid/
      images/
        image1.jpg
        image2.jpg
        ...
      labels/
        image1.txt
        image2.txt
        ...
```

Each label file should correspond to an image file, and the format of each label should be:

```
<class> <x_center> <y_center> <width> <height>
<class> <x_center> <y_center> <width> <height>
...
```
Each line corresponds to one bounding box in the image.

For further details on how to prepare your dataset, you can visit <https://docs.ultralytics.com/datasets/>.

For getting started quickly, you can navigate into the `examples` directory to download and partition a sample dataset.

## Step 5: Setting up configurations

All configuration is handled through the `client_config.yaml` file at the root of the repository. This single file controls:

- **Dataset paths** – Set `train`, `val`, and `test` to point to your local dataset splits.
- **Number of classes** – Set `nc` to the number of object classes.
- **Class names** – Define class names under `names`.
- **Training parameters** – Set `local_epochs` and `batch_size` to control local training.
- **Model architecture** – The `backbone` and `head` sections define the YOLOv8 model architecture.

Each client can have its own `client_config.yaml` with different dataset paths and training parameters to account for different hardware and data distributions.

##  Step 6: Building the compute package
Once you’ve completed all the configurations, build the compute package by running:
```bash
uv run scaleout package create -p client
```
This creates the compute package `package.tgz` which contains all the necessary files and configurations for the client environments.

If you make any changes to the model architecture or class configuration in `client_config.yaml`, you’ll need to rebuild (Step 6) and re-upload (Step 8) the compute package. For changes to training parameters only (`local_epochs`, `batch_size`, dataset paths), you don’t need to rebuild.

## Step 7: Initializing the seed model
To initialize the seed model, run the following command:
```bash
uv run scaleout run build -p client
```
This command will generate the seed model `seed.npz` that will be used as the starting point for the federated learning process.

## Step 8: Initializing the server-side
The next step is to initialize the server side. This is done by uploading the compute package and the seed model to the Scaleout Edge platform.
This is done by pressing the Sessions button and then the "New session" button. Here you can upload the compute package `package.tgz` and the seed model `seed.npz`.
Once the compute package and seed model are uploaded, you can create the session by pressing the "Create session" button. Here you also configure the total number of rounds and aggregator function for the federated learning process.

Before starting the training process, you need to connect the clients to the server which is done in the next step.

## Step 9: Connecting and starting the clients
To connect a client to the server, run the following command, replacing `<api_url>` with the API URL of your Scaleout Edge server:
```bash
uv run scaleout client start --api-url <api_url>
```
This starts the client and connects it to the server. Repeat this process for each client you want to connect to the server.


## Step 10: Training the global model
Once the clients are running, you can start the global training by pressing the Start session button in Scaleout Edge. This will initiate the federated learning process, where the global model is trained by aggregating the updates from the clients. Now you will see things happening on both the server and client side. You can monitor the training progress on the Scaleout Edge platform where metrics such as recall, precision, and mAP scores are shown.

<img src="figs/global_convergence.png" width=80% height=80%>


Once training is completed, you can download any model from a certain round in the session. The model can be used for inference on new data.

## Conclusion
In this tutorial, you have learned how to implement Ultralytics YOLOv8 models in a federated learning setting using the Scaleout Edge platform. By following the steps outlined in this tutorial, you have successfully set up a distributed training environment where clients independently train local models, and a global model is aggregated on the server. You have also learned how to configure the dataset, set up the model configurations, build the compute package, and start the federated learning process. By completing this tutorial, you have gained hands-on experience with federated learning and object detection tasks using Ultralytics models.

# Note
Steps 1, 2, 5, 6, 7 and 8 only need to be done once to set up the federated learning environment in Scaleout Edge.

To connect a new client, the only steps that need to be followed are steps 2, 3, 4 and 9.
Each client can have a different `client_config.yaml` to account for different hardware and training requirements.
