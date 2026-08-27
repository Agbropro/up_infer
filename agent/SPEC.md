i want to create code in python using simplest html script for user to evaluate model raw inference for computer vision models. For now it is for YOLO model with n number of class. 

for now let it handle the yolo type automatically then show the mask (if model is a segment model, and bbox if it's detection model). It chooses which model using a dropdown selection bar. with the name of the model and the path is configured in config/config.yaml

let the python uses ultralytics and fastapi if needed BE, it will process the image, infer it in domain, interfaces, applicatin with modular. for entites use typedict to ensure i know what the return type is. no argparse, simple docstring only one line. max func name is 3 word. make it easy to understand

let the html be in frontend folder and make it simple and easy to understand, so i can customize it for later. let the project name is UP Infer, Upload and infer images.

For inference it will choose the model and show all label first. the user can customize the label that wants to be shown in the inference, all labels is shown and user can delete the unwanted label. 
the user can control the confidence and IoU metrics.

there is a popup help page that is simple but readable that explain the confidence and IoU metrics (it is a question mark symbol from the customization data when clicked it'll open the popup)
it can infer singular images from uploaded data or many images if the user provides it. but no photo

it will show the inference below in a pages system that can be customized 2 till 5 columns for image, make it per inference block there are 2 columns, one is the inferred image, and the other is the confidence of each corresponding class. make each class differs in colors but readable.

Make the ui use the color of the pokemon wishiiwashi the fish pokemon

log the requirements.txt and for now use /opt/personal/.personal-venv for the venv