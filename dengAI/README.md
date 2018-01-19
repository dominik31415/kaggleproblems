This script is modelling and predicting outbreaks of dengue fever. It is based on weather data and previous outbreaks in San Juan, provided by  https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

### Pre-processing
loadData.py : reads records, fills in NANs by interpolating in between neighbouring values

### Model
NN_Dense1.py : a NN with one fully connected layer. accuracy is estimated by comparing its prediction to the test data, uses tensorflow to optimize its parameters and automatically stops optimization once the accuracy stops improving

### Training and results
main.jpynb : main script. used for refining the parameters of NN_Dense1 and produces final figures. currently it predicts the number of expected dengue cases four weeks in advance. For other prediction windows simply adjust the parameter "nForecast". The  ![figure below](https://github.com/dominik31415/machine-learning/blob/master/dengAI/comparison.png) summarizes the main result, comparing the projected number of cases (4 weeks ahead) with the observed number.

