from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def get_hedging_strategy(asset: str, limit: int = 50) -> dict:
    """
    Calculates proxy risk metrics for the asset and returns a Hedge Fund Strategy 
    recommendation based on recent volatility and momentum.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        db = client["eventoracle"]
        
        # Get recent structured features for the asset to gauge regime
        features = list(db["features"].find({"asset": asset}).sort("timestamp", -1).limit(10))
        
        if not features:
            # Fallback strategy if no ML features yet
            return {
                "asset": asset,
                "beta": "1.0",
                "volatility": "Moderate",
                "strategy_name": "Standard Diversification",
                "strategy_desc": "Not enough data to form a quantitative hedge. Maintain standard 60/40 portfolio diversification.",
                "tools": ["Bonds", "Gold"]
            }
            
        recent = features[0]
        
        # Extract features
        volatility = recent.get("rolling_volatility", 0.01) * 100 # %
        rsi = recent.get("rsi", 50)
        
        # Determine Beta proxy based on volatility
        # Typically, higher vol = higher beta
        beta = min(2.5, round((volatility / 1.5) + 0.5, 2))
        
        # Generate Hedge Fund Strategy logic
        if volatility > 2.5:
            regime = "High Volatility"
            strategy = "Long Volatility & Tail-Risk Hedging"
            desc = "Hedge funds typically deploy an Options Straddle or buy Out-Of-The-Money (OTM) Put options to protect against sudden downside. Delta-neutral strategies are preferred here to profit from swings rather than direction."
            tools = ["VIX Futures", "OTM Put Options", "Safe-Haven Currencies (CHF/USD)"]
        elif rsi > 70:
            regime = "Overbought / Momentum"
            strategy = "Momentum with Trailing Hedges"
            desc = "The asset is hot. Hedge funds will ride the trend but utilize Trailing Stop-Losses and sell Covered Calls to collect premium while the asset is near its peak."
            tools = ["Covered Calls", "Trailing Stops"]
        elif rsi < 30:
            regime = "Oversold / Distressed"
            strategy = "Contrarian Mean-Reversion"
            desc = "Hedge funds look to buy the dip (Statistical Arbitrage) while hedging with negatively correlated assets or shorting a weaker competitor in the same sector (Pairs Trading)."
            tools = ["Pairs Trading", "Sector Shorting"]
        else:
            regime = "Stable Trend"
            strategy = "Risk Parity & Beta Hedging"
            desc = "In stable markets, hedge funds adjust their Beta to match their risk appetite. If holding this asset long, they might short an index ETF to hedge out broader market risk (Alpha isolation)."
            tools = ["Index ETF Short", "Diversified Commodities"]

        volume_z = recent.get("volume_zscore", 0.0)
        sentiment = recent.get("avg_sentiment", 0.0)

        # 1. Advanced Metrics: Value at Risk (VaR) 95% Confidence
        # Basic parametric VaR = Volatility * Z-score (1.645 for 95%)
        var_95 = volatility * 1.645

        # 2. Institutional Flow Proxy
        if volume_z > 1.5 and rsi > 55:
            inst_flow = "Heavy Accumulation (Smart Money Buying)"
        elif volume_z > 1.5 and rsi < 45:
            inst_flow = "Heavy Distribution (Smart Money Selling)"
        elif volume_z < -1.0:
            inst_flow = "Low Liquidity (Retail Dominated)"
        else:
            inst_flow = "Neutral / Algorithmic Trading"

        # 3. Options Skew Bias Proxy
        if volatility > 2.0 and sentiment < -0.1:
            options_bias = "Severe Put Skew (Downside Protection Pricing)"
        elif volatility > 1.5 and sentiment > 0.2:
            options_bias = "Call Skew (Speculative Upside Pricing)"
        else:
            options_bias = "Balanced / Flat Skew"

        return {
            "asset": asset,
            "regime": regime,
            "beta": str(beta),
            "volatility": f"{volatility:.2f}%",
            "optimal_hedge_ratio": f"{min(0.9, beta * 0.4):.2f}",
            "var_95": f"{var_95:.2f}%",
            "institutional_flow": inst_flow,
            "options_bias": options_bias,
            "strategy_name": strategy,
            "strategy_desc": desc,
            "tools": tools,
            "timestamp": datetime.utcnow().isoformat()
        }

        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(get_hedging_strategy("NIFTY"))
