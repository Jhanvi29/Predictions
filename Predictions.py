import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
import dataprocessing
from datetime import datetime, timedelta
import pandas as pd


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size , dropout_rate = 0.2):
        super(MyModel, self).__init__()

        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(50, output_size)

    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        
        return x
    

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains a neural network model.

    Args:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader for the training dataset.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - num_epochs: Number of epochs to train the model.
    """
    for epoch in range(num_epochs):
        total_loss = 0

        # Iterate over batches of data from DataLoader
        for batch in train_loader:
            inputs, targets = batch

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, targets)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the total loss
            total_loss += loss.item()

        # Print out the loss averaged over the dataset
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss/len(train_loader.dataset)}")
    

def calculate_batch_losses(data_loader, model, criterion, plot_losses=False, plot_title='Loss Plot' , show = False):
    """
    Calculates the loss for each batch in a given dataset using the provided model and loss criterion.
    Optionally plots the batch losses.

    Args:
    - data_loader: DataLoader for the dataset (training or testing).
    - model: The trained model for making predictions.
    - criterion: The loss function used to calculate loss.
    - plot_losses: Boolean, if True, plots the batch losses.
    - plot_title: Title for the plot if plotting is enabled.

    Returns:
    - List of losses for each batch in the dataset.
    - Average loss over the dataset.
    """
    batch_losses = []
    total_loss = 0.0
    for inputs, targets in data_loader:
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        batch_losses.append(loss.item())
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)

    if plot_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(batch_losses, label='Batch Loss')
        plt.title(f'{plot_title}: {average_loss:.6f}')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.legend()
        if show:
            plt.show()

    return batch_losses, average_loss, plt

@torch.no_grad()
def graph_predictions(model, test_loader, scaler, plot_title='Predictions vs Actual', show = False):
    """
    Evaluate the model on the test dataset and plot predictions against actual values.

    Args:
    - model: Trained PyTorch model.
    - test_loader: DataLoader for the test dataset.
    - scaler: Scaler object used for un-normalizing predictions.
    - plot_title: Title for the plot.
    - return_metrics: If True, return evaluation metrics.

    Returns:
    - Optional: Any evaluation metrics if return_metrics is True.
    """
    model.eval()  # Set the model to inference mode

    predictions = []  # Track predictions
    actual = []       # Track the actual labels

    for inputs, targets in test_loader:
        # Forward pass for the batch
        batch_preds = model(inputs).cpu().numpy()  # Convert to numpy array
        batch_preds = scaler.inverse_transform(batch_preds)

        batch_actual = targets.cpu().numpy()  # Convert to numpy array
        batch_actual = scaler.inverse_transform(batch_actual)

        # Extend the lists
        predictions.extend(batch_preds.flatten())
        actual.extend(batch_actual.flatten())

    # Plot actuals vs predictions in scatter plot
    fig1 = plt.figure(figsize=(8, 8))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{plot_title} (Scatter Plot)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    if show:
        plt.show()

    # Line plot
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(actual, label='Actual Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title(f'{plot_title} (Line Plot)')
    plt.xlabel('Sample')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()
    
    return predictions , fig1 , fig2
def process_and_export_predictions(start_date, predictions, aggregation='mean'):
    """
    Process predictions and export to Excel.

    :param start_date: datetime, start date for the predictions
    :param predictions: list, list of prediction values
    :param aggregation: str, type of aggregation ('mean' or 'sum')
    :return: None
    """
    # Create a date range
    dates = [start_date + timedelta(days=i) for i in range(len(predictions))]

    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'Prediction': predictions})

    # Extract month and year
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Group and aggregate
    if aggregation == 'sum':
        monthly_data = df.groupby(['Year', 'Month'])['Prediction'].sum().reset_index()
    else:
        monthly_data = df.groupby(['Year', 'Month'])['Prediction'].mean().reset_index()

    # Export to Excel
    monthly_data.to_excel('monthly_predictions.xlsx', index=False)
    
    return monthly_data


def main():
    file_path = 'data_daily.csv'
    predict_loader, train_loader, test_loader, scaler = dataprocessing.process_dataframe(file_path)
    model = MyModel(3 , 100 , 1)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 10

    train_model(model , train_loader , criterion, optimizer , num_epochs)
    #train_model(model , test_loader , criterion , optimizer , num_epochs)
    train_batch_losses, train_loss , fig = calculate_batch_losses(train_loader, model, criterion, plot_losses=True, plot_title='Training Loss' , show=True)
    test_batch_losses, test_loss , fig = calculate_batch_losses(test_loader, model, criterion, plot_losses=True, plot_title='Testing Loss' , show=True)
    print(train_loss)
    print(test_loss)
    predictions , figure1 , figure2 = graph_predictions(model, predict_loader, scaler, plot_title='My Model Predictions' , show=True)
    print(len(predictions))
    start_date = datetime(2022, 1, 1)
    monthly_data = process_and_export_predictions(start_date , predictions)
    


if __name__ == "__main__":
    main()