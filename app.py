import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(page_title="AlphaPulse Risk Monitor", page_icon="📈", layout="wide")

st.title("📈 AlphaPulse: Financial Risk & Volatility Monitor")
st.markdown("### Interactive Monte Carlo Simulation & Portfolio Analytics")

# --- 2. SIDEBAR (USER INPUTS) ---
st.sidebar.header("🔧 Portfolio Settings")

# Default Tickers
default_tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'V', 'AMZN', 'KO', 'PFE', 'XOM', 'TSLA']
tickers = st.sidebar.multiselect("Select Assets for Portfolio:", default_tickers, default=default_tickers)

# Date Range
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Investment Amount
initial_investment = st.sidebar.number_input("Initial Investment Amount ($)", value=10000)

# Run Button
if st.sidebar.button("Run Analysis"):
    with st.spinner('Downloading Market Data & Running Simulations...'):
        
        # --- 3. DATA INGESTION ---
        try:
            # Download Data (Auto Adjust for Splits)
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
            
            # Handle standard 'Close' column issue or multi-level index
            if 'Close' in data.columns:
                 data = data['Close']
            
            # Forward Fill missing values
            data = data.ffill()
            
            # Calculate Daily Returns
            returns = data.pct_change().dropna()
            
            # --- 4. DASHBOARD TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Market Data", "🔥 Risk Heatmap", "🎲 Monte Carlo Simulation"])

            # --- TAB 1: RAW DATA ---
            with tab1:
                st.subheader("Historical Price Data")
                st.dataframe(data.tail())
                
                st.subheader("Daily Returns (%)")
                st.dataframe(returns.tail())
                
                # Simple Line Chart
                st.subheader("Price History")
                st.line_chart(data)

            # --- TAB 2: CORRELATION HEATMAP ---
            with tab2:
                st.subheader("Correlation Heatmap (Diversification Check)")
                
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
                st.pyplot(fig_corr)
                
                st.info("💡 Insight: Dark Red = Stocks move together (High Risk). Blue = Stocks balance each other out (Good Diversification).")

            # --- TAB 3: MONTE CARLO SIMULATION ---
            with tab3:
                st.subheader(f"Predicting Future Portfolio Value ({initial_investment}$ Investment)")
                
                # Simulation Parameters
                np.random.seed(42)
                num_simulations = 10000
                time_horizon = 252 # 1 Year
                
                # Run Simulation
                simulation_df = pd.DataFrame()
                last_prices = data.iloc[-1]
                daily_vol = returns.std()
                avg_daily_return = returns.mean()
                
                progress_bar = st.progress(0)
                
                # Optimization: Vectorized Simulation (Faster than Loop)
                # Note: Keeping the loop logic simple for readability as per previous steps, 
                # but wrapping it for Streamlit display.
                
                simulated_price_paths = np.zeros((time_horizon, num_simulations))
                
                # Logic: We simulate returns and apply them to the initial investment
                for i in range(num_simulations):
                    # Random returns
                    random_returns = np.random.normal(avg_daily_return, daily_vol, (time_horizon, len(tickers)))
                    
                    # Calculate path for this specific simulation
                    # We assume equal weight in portfolio for simplicity
                    portfolio_path = []
                    current_value = initial_investment
                    
                    # Create a composite daily return for the portfolio (Equal Weights)
                    portfolio_daily_returns = np.mean(random_returns, axis=1)
                    
                    for r in portfolio_daily_returns:
                        current_value = current_value * (1 + r)
                        portfolio_path.append(current_value)
                    
                    simulation_df[f"Sim {i}"] = portfolio_path
                    
                    if i % 100 == 0:
                        progress_bar.progress(i / num_simulations)
                
                progress_bar.progress(100)

                # Plot Spaghetti Chart
                fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
                ax_sim.plot(simulation_df, alpha=0.1, color='blue')
                ax_sim.set_title("1,000 Possible Futures (1 Year Horizon)")
                ax_sim.set_ylabel("Portfolio Value ($)")
                ax_sim.set_xlabel("Trading Days")
                st.pyplot(fig_sim)

                # --- VaR CALCULATION ---
                ending_values = simulation_df.iloc[-1]
                future_value_95 = np.percentile(ending_values, 5)
                VaR_95 = initial_investment - future_value_95

                # Display Metrics
                st.markdown("### 📝 Risk Analysis Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Initial Investment", f"${initial_investment:,.2f}")
                col2.metric("Worst Case (5th Percentile)", f"${future_value_95:,.2f}", delta_color="inverse")
                
                # Dynamic Logic for VaR text
                if VaR_95 > 0:
                    col3.metric("Value at Risk (VaR)", f"-${VaR_95:,.2f}", help="You could lose this amount.")
                    st.error(f"⚠️ **Risk Warning:** With 95% confidence, your maximum expected loss is **${VaR_95:,.2f}**.")
                else:
                    col3.metric("Value at Risk (VaR)", f"+${abs(VaR_95):,.2f}", help="Even in worst case, you profit.")
                    st.success(f"✅ **Safe Harbor:** Even in the worst 5% of scenarios, this portfolio is projected to grow by **${abs(VaR_95):,.2f}**.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Tip: Try removing tickers that might have delisted or checking your internet connection.")

else:
    st.info("👈 Please select stocks and click 'Run Analysis' in the sidebar.")