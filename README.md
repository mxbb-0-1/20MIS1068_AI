
# Trademark Class Prediction AI



## Objective

The primary objective of this project is to create an AI model that can accurately predict trademark classes based on goods and services descriptions entered by users. The model leverages data from the USPTO ID Manual to make these predictions.

## Task Breakdown

- [x] **Develop AI Model**: An AI model was developed using a custom fine-tuning approach with a pre-trained BERT model to classify trademark classes.
  - **Note**: The transformers.Trainer API was not used; instead, a custom training script was implemented.
- [x] **Model Logging**: Used WandB for monitoring and logging model metrics throughout the training process.
- [x] **Test with Diverse Inputs**: The model was tested with various inputs, including those not present in the original dataset.
- [x] **REST API Implementation**: A REST API was developed using FastAPI to allow other developers to input descriptions and receive recommended trademark classes.
- [x] **Dockerization**: The application was Dockerized for easy deployment and scalability.
- [x] **User Request Handling**:
  - Tracks the frequency of API calls by each user.
  - Limits each user to a maximum of 5 requests, after which an HTTP 429 status code is returned.
- [x] **Inference Time Logging**: The inference time for each request is recorded and logged.

## Training Script for Trademark Classification Model

This Python script trains a neural network model for trademark classification using PyTorch. The script is designed to be used in a supervised learning context, where the model learns to classify input data into various categories based on labeled examples.

# Training Script 
```python

num_epochs = 3
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Iterate over batches in the training data
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()  # Reset gradients for each batch

        # Forward pass: Compute model predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss between predictions and true labels
        loss = criterion(logits, labels)
        total_loss += loss.item()  # Accumulate loss

        # Backward pass: Compute gradients
        loss.backward()
        optimizer.step()  # Update model parameters

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions

    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "accuracy": accuracy
    })

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


```
### Video Demonstration

Watch the video for a step-by-step guide on how to run and test the Docker container:
[Watch the Video](https://drive.google.com/file/d/11J4Yx4_2rXCS_rE_tNsVhWbbo7IEOCzN/view?usp=sharing)


## Process

1. **Model Development**:
   - **Pre-trained BERT**: Utilized a pre-trained BERT model for sequence classification.
   - **Custom Fine-Tuning**: Fine-tuned the model using a custom function, ensuring compliance with the project requirement not to use the transformers.Trainer API.
   - **Model Monitoring**: Integrated WandB for detailed monitoring and logging of model performance metrics.

2. **REST API**:
   - Built with FastAPI to provide an endpoint for predicting trademark classes.
   - Integrated database logging to track and limit user requests, ensuring robust API management.

3. **Dockerization**:
   - Created a Docker image of the application for easy deployment.
   - Instructions for running the Docker container and testing the API are provided below.

## How to Test the Dockerized Application

1. **Download the Docker Image**:
   - Pull the Docker image from Docker Hub using the following command:
     ```
     docker pull mrblueberry43/trademark_classification_app
     ```

2. **Run the Docker Container**:
   - Start the Docker container with the following command:
     ```
     docker run -d -p 8000:8000 mrblueberry43/trademark_classification_app
     ```

3. **Install curl (if not already installed)**:
   - To interact with the API, install curl inside the Docker container (if necessary):
     ```
     apt-get update && apt-get install curl -y
     ```

4. **Test the API**:
   - Use the following curl command to test the API:
     ```
     curl -X POST http://0.0.0.0:8000/predict/ -H "Content-Type: application/json" -d "{\"description\": \"Solar-powered lawn mowers\", \"user_id\": \"user3\"}"
     ```
   - After 5 requests from the same user, the API will return an HTTP 429 status code, indicating that the request limit has been reached.


---

## Screenshots:
![image](https://github.com/user-attachments/assets/52cc90a9-80c0-4f7e-9ca5-9daeb3523794)
![image](https://github.com/user-attachments/assets/32372648-da03-4f3a-92f4-84a5968add05)
After 5 requests from the same user, the API is returning an HTTP 429 status code.

# Wandb Model Logging
![image](https://github.com/user-attachments/assets/1298368d-59bd-401a-9f2f-e425fec565dd)
![image](https://github.com/user-attachments/assets/efcb2fc9-aa7d-4c8d-bcc4-992a2b904d00)

# Model Testing
![image](https://github.com/user-attachments/assets/678a726f-0bd8-441f-b9d0-372b859f2909)
