# Welding Defect Object Detection Example

This is an example Scaleout Edge project that trains a YOLOv8n model on images of welds to classify them as "good", "bad", or "defected". The dataset is pre-labeled and can be accessed for free from Kaggle https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection. See a few examples below,

<img src="figs/fig1.jpg" width=30% height=30%>

<img src="figs/fig2.jpg" width=30% height=30%>

<img src="figs/fig3.jpg" width=30% height=30%>


This example is generalizable to many manufacturing and operations use cases, such as automatic optical inspection. The federated setup enables the organization to make use of available data in different factories and in different parts of the manufacturing process, without having to centralize the data.


## Step 1: Downloading the data

Download the dataset from the following link and extract it to the `datasets` directory:
https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection


## Step 2: Partitioning the data

To partition the data for each client, run the following command:

```bash
python3 partition_data.py welding-defect-detection <num_splits>
```
Replace `<num_splits>` with the number of clients you want to partition the data for.

This generates the dataset partitions in the 'datasets' directory.

## Step 3: Setting up the client_config.yaml

Inside the 'client' folder configure the 'client_config.yaml' in the following way:

```bash
train: datasets/split_name_X/train/images
val: datasets/split_name_X/valid/images
test: datasets/split_name_X/test/images

nc: 3

names:
  0: Bad Weld
  1: Good Weld
  2: Defect
```

## Step 4: Return to the root guide and follow the instructions from there
Now your dataset is ready and you have configured the global settings for the YOLOv8 model. Return to the root guide and follow the instructions from there to continue with the federated learning process.