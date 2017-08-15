This script is modelling and predicting outbreaks of dengue fever. It is based on weather data and previous outbreaks in San Juan, provided by  https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/


loadData.py

reads records, fills in NANs by interpolating in between neighbouring values


NN_Dense1.py

a NN with one fully connected layer. accuracy is estimated by comparing its prediction to the test data, uses tensorflow to optimize its parameters and automatically stops optimization once the accuracy stops improving


main.jpynb  

main script. used for refining the parameters of NN_Dense1 and produces final figures. currently it predicts the number of expected dengue cases four weeks in advance. For other prediction windows simply adjust the parameter "nForecast". 

