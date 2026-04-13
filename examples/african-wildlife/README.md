# African Wildlife Example

This is an example with data from Ultralytics where we classify buffalos, elephants, rhinos and zebras using a YOLOv8 model.

## Step 1: Downloading the data

Download the dataset from the following link and extract it to the `datasets` directory:
<https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip>


## Step 2: Partitioning the data

To partition the data for each client, run the following command:

```bash
python3 partition_data.py african-wildlife <num_splits>
```
Replace `<num_splits>` with the number of clients you want to partition the data for.

This generates the dataset partitions in the 'datasets' directory. These partitions needs to be distributed to the respective clients and renamed to 'split_X' instead of 'african-wildlife_split_X.

## Step 3: Setting up the client_config.yaml

Inside the 'client' folder configure the 'client_config.yaml' in the following way:

```bash
train: datasets/fed_dataset_split_X/train/images
val: datasets/fed_dataset_split_X/valid/images
test: datasets/fed_dataset_split_X/test/images

nc: 4

names:
  0: Buffalo
  1: Elephant
  2: Rhino
  3: Zebra

```

## Step 4: Return to the root guide and follow the instructions from there
Now your dataset is ready and you have configured the global settings for the YOLOv8 model. Return to the root guide and follow the instructions from there to continue with the federated learning process.
