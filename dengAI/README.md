# machine-learning
machine learning, neural networks

This script is trying to model and predict dengue fever outbreaks.
The data was provided by https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

Files:
loadData.py           reads the raw data, fills in NANs by interpolating in between neighbouring values

NN_Dense1.py          a simple NN with one fully connected layer. 
                      accuracy is estimated by comparing its prediction to the test data
                      uses tensorflow to optimize its parameters and automatically stops optimization 
                      once the accuracy stops improving

main.jpynb            main script. repeatedly refines parameters of NN_Dense1 and produces final figures
