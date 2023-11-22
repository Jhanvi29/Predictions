# Predictions
Fetch-Rewards

This repository host a simple predictive network. The model predicts sales of each month of 2022 when the sales of the month 2021 was given. The model has 100 hidden units with 3 input features and one output feature.

The requirements are written in requirements.txt file. The model runs well in the conda environment after installing them.
The website is 
```
https://predictions-agfs.onrender.com
```
In order to run just the predictions model do 
```
python Predictions.py
```

If you want to run the app, use 
```
streamlit run app.py
```
The app is hosted on streamlit and the webservices used are render. 
In order to use docker containeriszation, the image is hosted in the docker hub use 
```
docker pull jha29/jhanvi
docker run -p 8501:8501 jha29/jhanvi
Open local host URL: http://0.0.0.0:8501
```
