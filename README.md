# Stock-Market-Prediction-Model-Using-LSTM-GRU-Hybrid-Model
 Development of a hybrid LSTM-GRU machine learning model for stock market prediction in the money market industry. By amalgamating the strengths of LSTM and GRU as well as interpretability of traditional machine learning models, the hybrid model seeks to enhance the predictive accuracy of stock market forecast
The implementation of a hybrid Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) 
model for stock market prediction aimed to leverage the strengths of both architectures to capture the 
intricate temporal dependencies and patterns in stock price movements. The hybrid model was trained 
and evaluated on historical stock price data to forecast future prices. Here is a comprehensive analysis of 
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
4.3. Training Performance
During training, the hybrid model demonstrated robust convergence. The loss function, monitored 
through Mean Squared Error, consistently decreased over epochs, indicating effective learning. The 
addition of dropout layers and early stopping techniques mitigated overfitting, ensuring the model 
generalized well to unseen data.
4.4. Testing Performance
On the testing dataset, the hybrid LSTM-GRU model achieved the following results:
• MAE: 0.012
• MSE: 0.0004
• RMSE: 0.02
These metrics indicate a high level of accuracy in predicting stock prices. The RMSE, being low, suggests 
that the predicted prices were close to the actual values reflecting a strong correlation between the predicted 
and actual prices.
