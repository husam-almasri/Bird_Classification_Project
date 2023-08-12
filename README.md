### Bird Classification Project
This project aims to classify bird species using machine learning techniques.  
The classification model is built using the TensorFlow framework and the MobileNetV2 architecture.

#### Project Overview
The goal of this project is to develop a machine learning model that can accurately classify different bird species.  
The model takes images of birds as input and predicts the corresponding bird species.

#### Setup and Dependencies
To run this project, you need to have the following dependencies installed:  
- numpy  
- pandas  
- matplotlib  
- seaborn  
- tensorflow  
- keras  
- cv2

You can install these dependencies using the following command:

Copy the code below and paste into your terminal
```
pip install numpy pandas matplotlib seaborn tensorflow keras opencv-python
```
#### Project Structure  
The project consists of the following files:  
birbs.ipynb: This file contains the main code for training the bird classification model.  
valid.ipynb: This notebook is used for loading the trained model and performing predictions on unseen validation data.  
selected_df.csv: This file contains a subset of the dataset used for training and testing the model.  
valid_df.csv: This file contains a subset of the dataset used for validation.  
model.h5: This file contains the trained bird classification model.  

#### How to Use
Clone the repository to your local machine.  
Install the dependencies mentioned above.  
Open the folder in your preferred Python environment/IDE (vscode, conda..).  
Run the code to train the model.  
Once the model is trained, you can use it to classify bird species by providing images as input.  

#### Dataset
The dataset used for this project consists of images of different bird species.  
The images are organized into directories based on their corresponding bird species.  
The dataset is not included in this repository but can be obtained from a reliable source [kaggle link](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

#### Training the Model
To train the bird classification model, follow these steps:  
Prepare the dataset by organizing the images into directories based on their corresponding bird species (should already be done).  
Update the dataset variable in the code to specify the path to the dataset directory (important).  
Run the code in the birbs.ipynb file to train the model.  
After training, the model will be saved as model.h5.

#### Evaluating the Model
To evaluate the performance of the trained model, switch to the valid.ipynb notebook, the following metrics are used:  
Test Loss: The loss value obtained on the test dataset.  
Test Accuracy: The accuracy of the model on the test dataset.  
The classification report and confusion matrix are also generated to provide a detailed analysis of the model's performance.  

#### Additional Visualization
The code includes additional visualization techniques, such as displaying sample images from the dataset,  
plotting the model loss and accuracy during training, and generating a heatmap to visualize the areas of the image that contributed most to the model's prediction.

##### Note: This README.md file is intended to provide a high-level overview of the project to non-technical individuals, such as recruiters. For more detailed information and code documentation, please refer to the code files (birbs.ipynb) and comments within them.
