import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_receipt_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Convert 'Date' to datetime and set it as index
    data['Date'] = pd.to_datetime(data['# Date'])
    data.set_index('Date', inplace=True)
    data.drop('# Date', axis=1, inplace=True)

    # Resampling data for monthly and weekly trends
    monthly_data = data.resample('M').mean()
    data['DayOfWeek'] = data.index.dayofweek
    day_of_week_mean = data.groupby('DayOfWeek')['Receipt_Count'].mean()

    # Plotting the three requested visualizations
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Trend Over Time (Daily)
    axes[0].plot(data['Receipt_Count'], label='Daily Data')
    axes[0].set_title('Trend Over Time (Daily)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Receipt Count')

    # Monthly Trends
    axes[1].bar(monthly_data.index.month, monthly_data['Receipt_Count'])
    axes[1].set_title('Monthly Trends')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Receipt Count')
    axes[1].set_xticks(range(1, 13))

    # Weekly Trends
    axes[2].bar(day_of_week_mean.index, day_of_week_mean.values)
    axes[2].set_title('Weekly Trends')
    axes[2].set_xlabel('Day of the Week')
    axes[2].set_ylabel('Average Receipt Count')
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.tight_layout()
    return plt

class CustomDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        # Return the size of the dataset
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retrieve the X and y values at the given index
        return self.inputs[idx], self.outputs[idx]
    
def process_dataframe(filename):
    
    data = pd.read_csv(filename)
    # Convert the '# Date' column to a datetime object
    data['# Date'] = pd.to_datetime(data['# Date'])
    
    # Set the '# Date' column as the DataFrame index
    data.set_index('# Date', inplace=True)
    
    # Extract the year, month, day, and day of the week
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['day_of_week'] = data.index.dayofweek  # Monday=0, Sunday=6
    
    features = data[['month', 'day', 'day_of_week']]
    
    # Initializing the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fitting the scaler to the features and transforming them
    normalized_features = scaler.fit_transform(features)
    
    # Creating a new DataFrame with the normalized features
    x = pd.DataFrame(normalized_features, columns=['month', 'day', 'day_of_week'], index=data.index)
    
    # Display the first few rows of the normalized data
    nn_inputs = x[['month', 'day', 'day_of_week']]
    nn_inputs.head()
    
    normalized_outputs = scaler.fit_transform(data['Receipt_Count'].values.reshape(-1, 1))
    inputs = ['month','day','day_of_week']
    
    x = torch.tensor(nn_inputs[inputs].values,dtype=torch.float)
    #print(x)
    y = torch.tensor(normalized_outputs,dtype=torch.float)
    #print(y)
    dataset = CustomDataset(x, y)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create a DataLoader for the training set
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    
    # Create a DataLoader for the testing set
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)
    
    predict_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    
    return predict_loader , train_loader , test_loader , scaler

def print_dataloader(dataloader, num_batches=4):
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i}")
        print("Inputs:", inputs)
        print("Targets:", targets)
        print()

# Print the first 3 batches from the DataLoader

def main():
    file_path = 'data_daily.csv'
    plt_figure = analyze_receipt_data(file_path)
    plt_figure.show()
    predict_loader, train_loader, test_loader, scaler = process_dataframe(file_path)
    #print_dataloader(predict_loader)

if __name__ == "__main__":
    main()
    


    


