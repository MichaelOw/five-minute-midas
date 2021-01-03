# Five Minute Midas
## Overview
- Predicting profitable day trading positions using historical price data
- Independent End-to-End Machine Learning Project
- Read the Medium article (coming soon!)
- Try the [Web App Demo](https://five-minute-midas.herokuapp.com/)
<img src="data/demo/demo.gif" width="50%" height="50%">

## Features
- Data collection from Yahoo Finance (SQLite3)
- Data transformation, feature engineering (pandas)
- ML model training, tuning and tracking (scikit-learn, MLflow)
- ML model deployment: API and web app (Flask, Streamlit)

## Data Pipeline
![](data/demo/pipeline.png)

## Installation
- Use **requirements.txt** for the demo.
- Use **requirements_full.txt** for all scripts.

## Methodology
- Minute-level price data is extracted, and filtered to those with [Bullish RSI Divergence](https://www.google.com/search?q=bullish+rsi+divergence)
- These filtered points and their respective profit/loss outcomes are used to train an ML classifier
- With the trained model, we can try to predict future profit/loss outcomes

## Credits
- Price data extracted with the help of the [yfinance](https://github.com/ranaroussi/yfinance) library, created and maintained by [Ran Aroussi](https://github.com/ranaroussi) and other contributors

## Contact
[Michael Ow @ LinkedIn](https://www.linkedin.com/in/michael-ow/)
