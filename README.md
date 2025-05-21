# Paddy-disease-detection-using-mobilenetv3large-as-base-model
This is a project that i made for my undergraduate thesis. this project focus on creating a machine learning solution in detecting disease in rice leaves. 
I used MobilenetV3 Large as my base model and perform transfer learning technique. in the notebook i train my model twice, the initial training is for featuer extraction
and the second one is for fine tuning. the model achieve good results according to some metrics like accuracy, precision, recall, f1-score, confusion matrix and classification report.
i deploy the model to a streamlit web app for convenience and also due to time restriction.

## Instructions
### 1. Clone the repo

### 2. make sure that necessary files are in the same folder
for ease of use i suggest you to make sure that the model is in the same directory as the streamlit app python file

### 3. run the streamlit file
run the app using either command prompt inside the app folder, or use terminal in your IDE (e.g visual studio code)
run "streamlit run stapp.py" to run the app

![Sample](sample.jpg)
