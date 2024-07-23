""" This code sets up a Flask web server to process images received via HTTP
    POST requests. The images are processed using a pre-trained encoder and 
    TD3 agent to generate actions, which are then returned as JSON responses.
    The server and neural network models are implemented using PyTorch, and 
    image processing is handled using OpenCV and PIL.
    
    In the process_image method, log the time the image was received by the 
    server (image_receive_time) and the time the action was processed 
    (action_process_time)."""


import time
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask("Main") # Creates a new Flask web application named "Main"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32))

        self._out_features = (32, 3, 5)

    def forward(self, x):
        return self.features(x)

    def load_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path))

class Actor(nn.Module):
    def __init__(self, action_dim=2, max_action=1.0, encoder=None):
        super(Actor, self).__init__()
        self.encoder = encoder or Encoder()
        flat_size = 32 * 3 * 5
        self.actor = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1),
                                   nn.Flatten(),
                                   nn.Dropout(0.3),
                                   nn.Linear(flat_size, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, action_dim),
                                   nn.Sigmoid())

        self.max_action = max_action

    def forward(self, x):
        x = self.actor(x)
        print(f"Action shape from the actore network : {x.shape} , and the action is {x}`")
        return self.max_action * x

    def load_encoder_weights(self, weights):
        self.encoder.load_state_dict(weights, strict=False)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict["actor_state_dict"], strict=False)

class TD3Agent:
    def __init__(self, model_path , encoder_path):
        self.encoder = Encoder()
        self.encoder.load_weights(encoder_path)
        self.model = Actor(encoder=self.encoder) # ceate an instance of the Actor network
        self.model.load_weights(model_path) # load the weights of the model from the specified path
        self.model.to(device) # move the model to the device (CPU or GPU)
        self.model.eval() # set the model to evaluation mode (no training)

    def get_action(self, latent_vector):
        # Converts the latent vector to a PyTorch tensor, adds a batch dimension, and moves it to the specified device.
        latent_tensor = torch.from_numpy(latent_vector).float().to(device)    
        with torch.no_grad():
            action = self.model(latent_tensor)# Passes the latent tensor through the model to get the action.
        print(f"Action shape from the actore network : {action.shape} , and the action is {action}`")
        print(f"Action sreturn to the agent  : {action.action.squeeze().cpu().numpy()} ")
        return action.squeeze().cpu().numpy() # Removes the batch dimension and moves the tensor to the CPU.

class EncoderModel:
    def __init__(self, model_path):
        self.model = Encoder()
        self.model.load_weights(model_path)
        self.model.to(device)
        self.model.eval()
    
    # Checks the shape of the image and adjusts it to have 4 dimensions (batch, channel, height, width).
    def encode(self, image):
        # Ensure the image has the correct shape (adding batch and channel dimensions)
        if len(image.shape) == 2:
            # If it's a 2D array (height, width), add batch and channel dimensions
            image = image[np.newaxis, np.newaxis, :, :]
        elif len(image.shape) == 3:
            # If it's a 3D array (channel, height, width), add only batch dimension
            image = image[np.newaxis, :, :, :]
        
        #print(f"Image shape before encoding: {image.shape}")
        
        image_tensor = torch.from_numpy(image).float().to(device)
        with torch.no_grad():
            latent_vector = self.model(image_tensor)
        print(f"Latent vector: {latent_vector.cpu().numpy()} and returned  {latent_vector.squeeze(0).cpu().numpy()}")  # Add this line to debug the latent vector
        #print(f"Latent vector shape after encoding: {latent_vector.shape}")
        
        return latent_vector.squeeze(0).cpu().numpy()

encoder_model = EncoderModel('weights/encoder_weights1.pth')
td3_agent = TD3Agent('reinforcement/runs/TD3-20240717-171343/td3.pth', 'weights/encoder_weights1.pth')

@app.route("/process_image", methods=["POST"]) # Creates a new route "/process_image" that accepts POST requests only.  
def process_image(): # Defines a function that processes the image and returns the action.

    image_receive_time = time.time() # Time for receiving the image.
    print(f"Image received at server: {image_receive_time}")
    
    if 'file' not in request.files:
        print("No file part in the request")
        return "No file part", 400

    image_file = request.files['file']
    # Checks if the file part is empty and returns an error message if it is. 
    if image_file.filename == '':
        print("No selected file")
        return "No selected file", 400

    try:
        image = Image.open(BytesIO(image_file.read())) # Opens the image file and reads its content.
        image = np.array(image.convert('L')) # Converts the image to grayscale and then to a NumPy array. L = R * 299/1000 + G * 587/1000 + B * 114/1000
        image = cv2.resize(image, (80, 60)) # Resizes the image to 80x60 pixels.
        
        #print(f"Image shape after processing: {image.shape}")
        #latent_vector = encoder_model.encode(image)
        #action = td3_agent.get_action(latent_vector)

        latent_vectors=[]
        for i in range(4):
            latent_vector = encoder_model.encode(image)
            print(f"Latent vector shape after encoding: {latent_vector.shape}")
            breakpoint()
            latent_vectors.append(latent_vector)

        latent_vectors = np.stack(latent_vectors , axis=0) # Simulate accumulation of 4 vectors

        #latent_vectors = latent_vectors.transpose(1,0,2,3).reshape(1,-1,3,5) # reshape to (1,128,3,5)
        
        latent_vectors = np.stack(latent_vectors, axis=0)  # Shape: (4, 32, 3, 5)
        latent_vectors = latent_vectors.transpose(1, 0, 2, 3).reshape(1, 128, 3, 5)
        
        #print(f"Final latent vectors shape: {latent_vectors.shape}")
        action = td3_agent.get_action(latent_vectors)
        print(f"Action shape: {action.shape}, action: {action}")
        #action = td3_agent.get_action(concatenated_vector)
        action_process_time = time.time()
        #print(f"Action processed at server: {action_process_time}")

        total_delay_on_server = action_process_time - image_receive_time
        print(f"************Total delay: {total_delay_on_server} seconds ************")

        response_data = {"vel_left": float(action[0]), "vel_right": float(action[1])}
        print (f"The action space is {action[0]} on the left wheel and  {action[1]} on the right wheel ****  {action[0], action[1]}****")

        return jsonify(response_data), 200 # Returns the action as a JSON response with status code 200.
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}", 400

if __name__ == "__main__":
    app.run(host="192.168.2.25", port=5000, debug=True) # for mars bot it is host="192.168.75.248"  192.168.2.25
