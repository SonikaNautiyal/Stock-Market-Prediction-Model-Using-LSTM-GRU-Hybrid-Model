# Stock-Market-Prediction-Model-Using-LSTM-GRU-Hybrid-Model
 Development of a hybrid LSTM-GRU machine learning model for stock market prediction in the money market industry. By amalgamating the strengths of LSTM and GRU as well as interpretability of traditional machine learning models, the hybrid model seeks to enhance the predictive accuracy of stock market forecast
Here is a comprehensive analysis of 
the results obtained:

The dataset comprised monthly stock prices over a span of 66 years, which was split into training and 
testing sets with an 80-20 ratio. Features such as open, high, low, and close prices, and trading volume 
were used. The data was normalized to a scale between 0 and 1 to facilitate efficient training. The hybrid 
model was configured with an initial LSTM layer followed by a GRU layer, both consisting of 50 units. 
This architecture aimed to exploit LSTM's ability to capture long-term dependencies and GRU's 
computational efficiency

 
The model's performance was assessed using several metrics:

• Mean Absolute Error (MAE)
• Mean Squared Error (MSE)
• Root Mean Squared Error (RMSE)


Training Performance :  

During training, the hybrid model demonstrated robust convergence. The loss function, monitored 
through Mean Squared Error, consistently decreased over epochs, indicating effective learning. The 
addition of dropout layers and early stopping techniques mitigated overfitting, ensuring the model 
generalized well to unseen data.
