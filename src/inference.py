import torch
import cv2
import numpy as np
from model import ResnetsegmentationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the model
model = ResnetsegmentationModel(num_classes=2).to(device)
model.load_state_dict(torch.load('flood_model.pth', map_location=device))
model.eval()

#load and process the image
image = cv2.imread('data/test/images/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
input_tensor = torch .tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Save the mask
cv2.imwrite("predicted_mask.png", mask * 255)