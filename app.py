import streamlit as st
import dataprocessing
import Predictions
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

st.title('Prediction Model Interface')

st.header('Introduction')
st.write("This app hosts a prediction model that predicts the sale of the receipts in the year 2022 given the sales of 2021.")

st.header("Data Exploration")
st.write("In order to make a decision on which model to deploy data exploration is performed. TThe data given is a time series, therefore season decomposition is performed ")
st.header("Plots for data analysis")
plot = dataprocessing.analyze_receipt_data('data_daily.csv')
st.pyplot(plot)

st.write("The analysis of the receipt data uncovers distinct temporal trends." 
        "Daily trends show significant fluctuations, likely influenced by factors like holidays, promotions, or seasonal changes, evidenced by peaks and troughs throughout the year. "
        "Monthly analysis reveals clear seasonal trends, with certain months experiencing higher activity, possibly due to consumer shopping patterns or seasonal events." 
        "The weekly pattern analysis indicates distinct differences between weekdays and weekends, suggesting variations in consumer engagement." 
        "These insights are crucial for understanding consumer behavior, aiding in strategic planning for inventory, staffing, and marketing in sectors sensitive to temporal changes")

st.header("Feature Engineering")
st.write("The data provided only provides sales which unfortunately does not provide the best prediction results. Therefore to make the" 
         "infromation richer, more features namely ,  'month' , 'day' , 'day_of_week',  have been added to account for seasonal changes."
         "We add these as our features because the above graph gives a clear indication that the data is dependent on these features")



st.header("Data Preperation")
st.write("The data pre-processing is an important feature of predicting tasks. The data provided has sales in millions. A neural netwok works best when the inputs are in 0 and 1."
         "Therefore the data is normalised. The normalisation technique just makes the data in between 0 and 1. In order to do that python's inbuilt functions are used." )

st.header('Model Infromation')
st.write("The model used for prediction is a simple stacking up of linear layers. Since the amount of data is small we have to be careful about the overfitting so in between the linear layers, regularisation layers have been added"
         "In all neural network activation is of utmost importance so for activating the netwroks, rectified linear unit are stacked up with linear layer")


st.header("Training")
st.write("The training is done my making custom dataset of the inputs of the neural network which are the features."
         "The mean squared error loss is calculated between the sales given in the dataset and the output of the neural netwrok"
         "The model works best when trained with 25 epochs but feel free to experiement with number of epochs. Althogh, the model can"
         "overfit if the number of epochs is greater than 100")


# User input for number of epochs

st.header("Model Initialisation")
st.write("Enter number of inputs , hidden units and number of outputs and dropout rate. In our case number of inputs is 3 , output is 1")
st.write("Dropout rates are added so that the model does not overfit and the best way to do is to add dropout layers in the model."
         "The model works best when the probability of dropout is 0.2")
st.write("Hidden units is something that can be toyed aroung with. Model gives best output with hidden units as 100")

number_of_inputs = st.number_input('Enter the size of the input' , min_value = 3 , max_value = 3 , value = 3)
hidden_units  = st.number_input('Enter the number of neurons in hidden unit' , min_value=10 , max_value=200 , value = 100)
output_size = st.number_input('Enter the size of the output' , min_value=1 , max_value=1 , value = 1)
drop_prob = st.number_input('Enter the dropout probability between 0 and 1 ' , min_value=0.0 , max_value=1.0 , value= 0.2)

model = Predictions.MyModel(number_of_inputs , hidden_units , output_size , dropout_rate = drop_prob)



num_epochs = st.number_input('Enter the number of epochs', min_value=1, max_value=100, value=10)


if st.button('Run Model'):
    predict_loader, train_loader, test_loader, scaler = dataprocessing.process_dataframe('data_daily.csv')
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    Predictions.train_model(model, train_loader, criterion, optimizer, num_epochs)
       
    
    st.header("Training Loss Graph")
    # Ensure a new figure is created for the training loss graph
    plt.figure()
    train_batch_losses, train_loss , fig = Predictions.calculate_batch_losses(train_loader, model, criterion, plot_losses=True, plot_title='Training Loss' , show=False)
    last_5_losses = train_batch_losses[-5:]
    st.write("Train Batch Losses:", last_5_losses)
    #st.write("Train Batch Losses:", train_batch_losses)
    st.write("Total Train Loss:", train_loss)
    st.pyplot(fig)

    
    st.header("Testing")
    st.write("The testing is done by dividing the dataset into two parts. The training contains 90 percent of the dataset and and 10 percent"
             "testing. As we can infer from the graph that the training loss is lesser than the testing loss. This is very common scenario."
             "Since the difference is not vast, this just tells us that the model works well on the data that it hasnt been trained on. "
             "This inference shows that the model hasn't overfitted and the predicted values would be close to the real ones.")
    plt.figure()
    test_batch_losses, test_loss , figure = Predictions.calculate_batch_losses(test_loader, model, criterion, plot_losses=True, plot_title='Testing Loss' , show=False)
    last_5_losses = test_batch_losses[-5:]
    st.write("Train Batch Losses:", last_5_losses)
    st.write("Total Train Loss:", test_loss)
    st.pyplot(figure)


    st.header("Prediction")
    st.write("The model is used to predict the values. Since the predeicted values are between 0 and 1, they are converted to the units "
             "as given in the dataset be perfoming inverse if the scalar.")
    
    predictions , scatter_plot , line_plot = Predictions.graph_predictions(model, predict_loader, scaler, plot_title='My Model Predictions' , show=True)
    st.pyplot(scatter_plot)
    st.pyplot(line_plot)

    # Generate monthly data
    st.write("The predicted value and actual values are sort of close to each other and we can say that this model is stable and predicts almost close to the real value")
    st.header("Displaying Sales")
    start_date = datetime(2022, 1, 1)
    monthly_data = Predictions.process_and_export_predictions(start_date , predictions)

    # Display monthly data
    st.write("Monthly Aggregated Predictions:")
    st.dataframe(monthly_data)