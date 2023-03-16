import pandas as pd
import numpy as np
import scipy.optimize as sco
import streamlit as st


# Define functions for portfolio optimization
def calculate_portfolio_risk(weights, correlation_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(correlation_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility

def calculate_portfolio_return(weights, portfolio_returns):
    portfolio_return = np.dot(weights.T, portfolio_returns.mean().values * 252)
    return -portfolio_return

# Define Streamlit app
def app():
    # Create sidebar
    st.sidebar.header('Portfolio Optimization Inputs')
    
    # Get user inputs
    symbols = st.sidebar.text_input('Enter symbols separated by commas (e.g. AAPL,MSFT,GOOG)', 'AAPL,MSFT,GOOG')
    symbols = [s.strip().upper() for s in symbols.split(',')]
    allocation = st.sidebar.slider('Allocation %', 0, 100, 20)
    risk_tolerance = st.sidebar.slider('Risk Tolerance', 0.0, 1.0, 0.1, step=0.05)
    return_target = st.sidebar.slider('Return Target', 0.0, 1.0, 0.2, step=0.05)
    
    # Fetch data
    df_prices = pd.DataFrame()
    for symbol in symbols:
        df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?interval=1d&events=history&includeAdjustedClose=true',
                         index_col='Date', parse_dates=['Date'])
        df_prices[symbol] = df['Adj Close']
    df_returns = df_prices.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = df_returns.corr()
    
    # Define optimization constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: calculate_portfolio_return(x, df_returns) - return_target},
                   {'type': 'ineq', 'fun': lambda x: risk_tolerance - calculate_portfolio_risk(x, correlation_matrix)},
                   {'type': 'ineq', 'fun': lambda x: x}]
    
    # Define initial weights
    weights = np.ones(len(symbols)) / len(symbols)
    
    # Perform optimization
    result = sco.minimize(calculate_portfolio_risk, weights, method='SLSQP', constraints=constraints)
    optimized_weights = result.x.round(2)
    
    # Display optimized portfolio
    st.header('Optimized Portfolio')
    for symbol, weight in zip(symbols, optimized_weights):
        st.write(f"{symbol}: {weight*allocation}%")
