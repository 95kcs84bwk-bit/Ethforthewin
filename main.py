import json
import sys
from eth_bot import ETHTradingBot
from backtester import ETHBacktester
from dashboard import TradingDashboard
from risk_monitor import RiskMonitor
import threading

def main():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║            ETH/USD Algorithmic Trading Bot           ║
    ╚══════════════════════════════════════════════════════╝
    
    1. Run Backtest
    2. Start Live Trading
    3. Launch Dashboard
    4. Risk Monitor Only
    5. Exit
    
    """)
    
    choice = input("Select option (1-5): ")
    
    if choice == '1':
        # Run backtest
        print("\nRunning backtest...")
        backtester = ETHBacktester('data/eth_historical.csv')
        results = backtester.run_backtest()
        backtester.generate_report()
        
    elif choice == '2':
        # Start live trading
        print("\nStarting live trading bot...")
        bot = ETHTradingBot(config)
        
        # Start risk monitor in separate thread
        risk_monitor = RiskMonitor(bot)
        monitor_thread = threading.Thread(
            target=risk_monitor.run_checks,
            daemon=True
        )
        monitor_thread.start()
        
        # Start trading bot
        bot.run()
        
    elif choice == '3':
        # Launch dashboard
        print("\nLaunching dashboard on http://localhost:8050")
        bot = ETHTradingBot(config)
        dashboard = TradingDashboard(bot)
        dashboard.run()
        
    elif choice == '4':
        # Risk monitor only
        print("\nStarting risk monitor...")
        bot = ETHTradingBot(config)
        risk_monitor = RiskMonitor(bot)
        
        while True:
            risk_monitor.run_checks()
            time.sleep(60)  # Check every minute
    
    else:
        print("Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
