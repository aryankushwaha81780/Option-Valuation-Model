import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- Configuration Parameters ---
TICKER = "NVDA"
TARGET_EXPIRY = "2026-02-20"  # February monthly expiration
STRIKE_PRICE = 185.00
RISK_FREE_RATE = 0.0372       # 3.72% based on 1-Month Treasury
ANALYSIS_DATE = "2026-01-11"  # Current date per prompt context

class OptionPricingModel:
    def __init__(self, ticker, spot_price, strike, time_to_expiry, risk_free_rate):
        self.ticker = ticker
        self.S = spot_price
        self.K = strike
        self.T = time_to_expiry
        self.r = risk_free_rate
    
    def calculate_d1_d2(self, sigma):
        if sigma <= 1e-6: sigma = 1e-6
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        return d1, d2

    def bsm_price(self, sigma, option_type='call'):
        d1, d2 = self.calculate_d1_d2(sigma)
        if option_type == 'call':
            price = (self.S * si.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2))
        elif option_type == 'put':
            price = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2) - self.S * si.norm.cdf(-d1))
        return price

    def calculate_greeks(self, sigma, option_type='call'):
        d1, d2 = self.calculate_d1_d2(sigma)
        delta = si.norm.cdf(d1) if option_type == 'call' else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T) * si.norm.pdf(d1) * 0.01 
        theta_term1 = -(self.S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            theta = theta_term1 - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)
        else:
            theta = theta_term1 + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2)
            
        theta = theta / 365.0 # Daily decay
        return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta}

    def implied_volatility(self, market_price, option_type='call', max_iterations=100):
        sigma = 0.5
        for i in range(max_iterations):
            price = self.bsm_price(sigma, option_type)
            d1, d2 = self.calculate_d1_d2(sigma)
            vega = self.S * np.sqrt(self.T) * si.norm.pdf(d1)
            diff = market_price - price
            if abs(diff) < 1e-5: return sigma
            if vega == 0: return np.nan
            sigma = sigma + diff / vega
        return np.nan

    def generate_hedging_advice(self, sigma, contract_size=100):
        greeks = self.calculate_greeks(sigma)
        delta = greeks['Delta']
        
        action = "SHORT" if delta > 0 else "LONG"
        hedge_qty = abs(delta * contract_size)
        
        print(f"\n--- HEDGING STRATEGY (Delta Neutral) ---")
        print(f"Current Delta: {delta:.4f}")
        print(f"Risk Exposure: Holding 1 Option Contract behaves like holding {delta*100:.1f} shares of {self.ticker}.")
        print(f"RECOMMENDATION: To hedge this position, the fund should {action} {hedge_qty:.0f} shares of {self.ticker}.")
        print(f"Note: This hedge must be rebalanced as the stock price moves (Gamma Risk).")

    # --- STEP 3: Sensitivity Matrix ---
    def sensitivity_analysis(self, base_sigma):
        print("\n--- SENSITIVITY ANALYSIS (Scenario Matrix) ---")
        # Define scenarios: Spot +/- 10%, Vol +/- 20%
        spots = [self.S * 0.9, self.S * 0.95, self.S, self.S * 1.05, self.S * 1.1]
        vols = [base_sigma * 0.8, base_sigma * 0.9, base_sigma, base_sigma * 1.1, base_sigma * 1.2]
        
        # Create labels
        col_labels = [f"Spot ${s:.2f}" for s in spots]
        idx_labels = [f"Vol {v:.1%}" for v in vols]
        
        matrix = []
        for vol in vols:
            row = []
            for s in spots:
                # Create a temporary model state for the scenario
                # We assume T and r stay constant for this instantaneous snapshot
                temp_model = OptionPricingModel(self.ticker, s, self.K, self.T, self.r)
                price = temp_model.bsm_price(vol, 'call')
                row.append(price)
            matrix.append(row)
            
        df_sens = pd.DataFrame(matrix, columns=col_labels, index=idx_labels)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_sens)
        print("\nInterpretation: This table shows estimated Option Price under different market conditions.")

def bsm_price_dynamic(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return (S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))
    return (K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))

def generate_stock_path(S0, mu, sigma, days, dt, start_date_str):
    '''
    1. Set a Fixed Seed for Stability
    We use the integer value of the starting price as the seed
    This ensures "randomness" is consistent across runs for this specific stock
    '''
    np.random.seed(int(S0)) 
    
    prices = [S0]
    dates = [datetime.strptime(start_date_str, "%Y-%m-%d")]
    current_price = S0
    
    # Pre-generate shocks with a fixed seed
    shocks = np.random.normal(0, np.sqrt(dt), days)
    
    for i in range(days):
        shock = shocks[i]
        
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * shock
        
        current_price = current_price * np.exp(drift + diffusion)
        
        prices.append(current_price)
        dates.append(dates[-1] + timedelta(days=1))
        
    return np.array(prices), dates

def run_analysis_and_simulation():
    expiry_dt = datetime.strptime(TARGET_EXPIRY, "%Y-%m-%d")
    analysis_dt = datetime.strptime(ANALYSIS_DATE, "%Y-%m-%d")
    days_to_expiry = (expiry_dt - analysis_dt).days
    
    if days_to_expiry <= 0:
        print("Error: Option has already expired.")
        return

    SIMULATION_DAYS = days_to_expiry + 5
    T_years = days_to_expiry / 365.0
    DT = 1/365.0

    print(f"--- ANALYZING {TICKER} ---")
    print(f"Analysis Date: {ANALYSIS_DATE}")
    print(f"Expiry Date:   {TARGET_EXPIRY} ({days_to_expiry} days)")

    start_date = (analysis_dt - timedelta(days=400)).strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date} to {ANALYSIS_DATE}...")
    df = yf.download(TICKER, start=start_date, end=ANALYSIS_DATE, progress=False)
    
    if df.empty:
        print(f"CRITICAL ERROR: No data found for {TICKER}. Check your internet or Ticker symbol.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1) 
        except:
            pass

    if 'Close' in df.columns:
        spot_data = df['Close']
    else:
        spot_data = df.iloc[:, 0]
        
    spot_data = pd.to_numeric(spot_data, errors='coerce').dropna()
    
    current_spot = float(spot_data.iloc[-1])
    
    log_returns = np.log(spot_data / spot_data.shift(1)).dropna()
    
    if len(log_returns) < 20:
        print("Error: Not enough data points to calculate volatility.")
        return

    hv_252 = log_returns.rolling(window=252).std().iloc[-1] * np.sqrt(252)
    
    if np.isnan(hv_252):
        hv_252 = log_returns.std() * np.sqrt(252)

    model = OptionPricingModel(TICKER, current_spot, STRIKE_PRICE, T_years, RISK_FREE_RATE)
    price_model = model.bsm_price(hv_252, 'call')
    
    print(f"\nSpot Price: ${current_spot:.2f}")
    print(f"Volatility (Annualized): {hv_252:.2%}")
    print(f"Theoretical Option Price: ${price_model:.2f}")

    model.generate_hedging_advice(hv_252)

    model.sensitivity_analysis(hv_252)

    print(f"\n--- Generating {SIMULATION_DAYS}-Day Simulation to Expiry ---")
    stock_prices, sim_dates = generate_stock_path(current_spot, RISK_FREE_RATE, hv_252, SIMULATION_DAYS, DT, ANALYSIS_DATE)
    
    option_prices = []
    intrinsic_values = []
    
    for i, s_t in enumerate(stock_prices):
        curr_date = sim_dates[i]
        days_remaining = (expiry_dt - curr_date).days
        t_remain = max(0, days_remaining / 365.0)
        
        opt_p = bsm_price_dynamic(s_t, STRIKE_PRICE, t_remain, RISK_FREE_RATE, hv_252, 'call')
        option_prices.append(opt_p)
        intrinsic_values.append(max(0, s_t - STRIKE_PRICE))

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1)
    
    # PLOT 1: REAL HISTORICAL DATA
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(spot_data.index, spot_data.values, color='black', label=f'Actual {TICKER} History')
    ax0.axhline(y=STRIKE_PRICE, color='red', linestyle='--', alpha=0.6, label='Strike Price')
    ax0.set_title(f"PART 1: Real Market Context - {TICKER} Past 1 Year", fontweight='bold')
    ax0.set_ylabel("Stock Price ($)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # PLOT 2: SIMULATED FUTURE STOCK PATH
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(sim_dates, stock_prices, color='navy', label='Projected Path (Monte Carlo)')
    ax1.axhline(y=STRIKE_PRICE, color='red', linestyle='--', alpha=0.6, label='Strike Price')
    ax1.axvline(x=expiry_dt, color='orange', linestyle=':', label='Expiry Date')
    ax1.set_title(f"PART 2: Simulated Future Movement (Next {SIMULATION_DAYS} Days)", fontweight='bold')
    ax1.set_ylabel("Projected Stock Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PLOT 3: OPTION PRICE CONVERGENCE
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(sim_dates, option_prices, color='green', linewidth=2, label='BSM Option Price')
    ax2.fill_between(sim_dates, intrinsic_values, color='gray', alpha=0.3, label='Intrinsic Value')
    ax2.axvline(x=expiry_dt, color='orange', linestyle=':', label='Expiry Date')
    ax2.set_title("PART 3: Option Valuation & Time Decay", fontweight='bold')
    ax2.set_ylabel("Option Price ($)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_analysis_and_simulation()