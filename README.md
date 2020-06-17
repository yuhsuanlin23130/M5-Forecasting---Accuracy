
Kaggle Competition: M5-Accuracy, M5-Uncertainty

##### M5-Accuracy (https://www.kaggle.com/c/m5-forecasting-accuracy)
預測Walmart各商品(30490 in total)未來28天之銷售量。
Model: One to one LSTM。
根據觀察，銷售量有週期性的變化，而LSTM擅長這種時間序列的處理，且比起RNN又多了input gate、forget gate、output gate三個gates強化學習效果，適合用於此銷售量的預測。
經過嘗試後發現單純只用"sales_train_validation.csv"中的銷售量資訊做為training data去訓練模型就可得到不錯的結果，嘗試加上Feature Engineering、”calender.csv”提供之資訊並沒有顯著提升結果。
Best score on Kaggle: 0.7410。

preprocessing
- startDay = 300: 前300天銷售資料不考慮，預測結果可稍微提升
- 取d_301 ~ d_1913作為training dataset
- timesteps = 21: 訓練階段取前21天的銷售量為X，下一天銷售量為y。預測階段，首先使用最近的21天(timesteps)來預測下一天(d_1914)的銷售量。接著將siding window往後移動一格，以已知的20天和預測的d_1914作為input (X)預測d_1915銷售量，以此類推。
- Feature Scaling by MinMaxScaler(feature_range = (0, 1))

LSTM Model
(After tuning the parameters)
- 4 hidden layers, with 50, 200, 400, 800 nodes respectively.
- dropout=0.2 in each layer
- optimizer='adam'
- loss='mean_squared_error'
- learning rate=0.01


##### M5-Uncertainty  (https://www.kaggle.com/c/m5-forecasting-uncertainty)
預測Walmart各商品(30490 in total)未來28天之銷售量分布，即預測分布之percentile (u1=0.005, u2=0.025, u3=0.165, u4=0.25, u5=0.5, u6=0.75, u7=0.835, u8=0.975, and u9=0.995)。
Repeat LSTM model 100 times for each item.  Use the item’s 100 results at day x as the distribution of that item at day x. 
The file ./Uncertainty.ipynb get use of the 100 files resulted from the M5-Accuracy problem, compute the percentiles of each item and convert the results to the format that kaggle accepts.
