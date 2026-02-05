#!/bin/bash

# ETH Trading Bot Setup Script
echo "Setting up ETH Trading Bot..."

# Create virtual environment
python3 -m venv eth_bot_env
source eth_bot_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install ccxt pandas numpy TA-Lib scikit-learn vectorbt matplotlib

# Install TA-Lib (system dependency)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    pip install TA-Lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ta-lib
    pip install TA-Lib
fi

# Create directory structure
mkdir -p logs data backups config

# Create configuration file
cat > config/config.json << 'EOF'
{
  "exchange": "binance",
  "api_key": "YOUR_API_KEY_HERE",
  "api_secret": "YOUR_API_SECRET_HERE",
  "symbol": "ETH/USDT",
  "use_futures": true,
  "initial_balance": 10000,
  "risk_per_trade": 0.01,
  "max_daily_drawdown": 0.02,
  "max_daily_trades": 5,
  "max_consecutive_losses": 4,
  "min_confidence": 0.65,
  "max_position_size_usd": 1000,
  "stop_loss_atr_multiplier": 2.2,
  "take_profit_1_rr": 1.5,
  "take_profit_2_rr": 3.0,
  "trailing_stop_atr": 20,
  "optimal_trading_hours": [13, 14, 15, 16, 17, 18, 19, 20, 21],
  "enable_ml_confirmation": false,
  "whale_alert_threshold": 10000,
  "log_level": "INFO",
  "paper_trading": true
}
EOF

echo "Setup complete! Edit config/config.json with your API keys."
echo "To run: source eth_bot_env/bin/activate && python eth_bot.py"
