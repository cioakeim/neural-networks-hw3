# Neural Networks 2024 - HW Assignment 3

## Dependencies:
AMD's AOCL is used as an algebra backend for this project. For algebra operations Eigen is used.
For pcaTest, OpenCV is used for image display.

## Usage:
There are 2 basic scripts that are in use:
./pcaTest : Displays 10 images in the original and recreated form using PCA with 124 components.

./autoEncoderTest [-d | config_file_path]
Trains an autoencoder and logs the train and test losses at each epoch. The config_file_path parameter defines where the configuration
file path is. It contains 3 .txt files defining the config of the general operation, the autoencoder parameters and the optimizer parameters.

Using -d the default config file path is used.

Without arguments, the precompiled default variables in the script are used.
