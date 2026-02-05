import pandas as pd
import numpy as np
from datetime import datetime
import vectorbt as vbt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ETHBacktester:
    def __init__(self, data_path='eth_data.csv'):
        self.data = pd.read_csv(data_path, parse_dates=['timestamp'])
        self.results = {}
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Technical features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_change'] = df['volume'].pct_change()
        
        # Price action features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Market regime features
        df['trend_strength'] = df['close'].rolling(50).apply(
            lambda x: (x[-1] - x[0]) / x[0]
        )
        
        return df.dropna()
    
    def train_ml_model(self, df):
        """Train Random Forest model for entry confirmation"""
        features = [
            'returns', 'volatility', 'volume_change', 'high_low_ratio',
            'close_open_ratio', 'body_size', 'upper_shadow', 'lower_shadow',
            'trend_strength', 'rsi', 'macd', 'macd_hist'
        ]
        
        # Create target (1 for profitable trades, 0 otherwise)
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.01).astype(int)
        
        X = df[features].fillna(0)
        y = df['target'].fillna(0)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"ML Model - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        return model, scaler, features
    
    def run_backtest(self, start_date='2022-01-01', end_date='2024-01-01'):
        """Run comprehensive backtest"""
        # Filter data
        mask = (self.data['timestamp'] >= start_date) & (self.data['timestamp'] <= end_date)
        df = self.data[mask].copy()
        
        # Calculate all indicators
        df = self.prepare_features(df)
        
        # Train ML model
        ml_model, scaler, features = self.train_ml_model(df)
        
        # Generate signals based on strategy
        df['signal'] = 0
        
        # Strategy logic
        for i in range(200, len(df)):
            # Trend filter (using EMA 200)
            trend_up = df['close'].iloc[i] > df['ema_200'].iloc[i]
            trend_down = df['close'].iloc[i] < df['ema_200'].iloc[i]
            adx_strong = df['adx'].iloc[i] > 25
            
            # Entry conditions
            if trend_up and adx_strong:
                # Long conditions
                price_near_ema = abs(df['close'].iloc[i] - df['ema_50'].iloc[i]) / df['ema_50'].iloc[i] < 0.005
                rsi_ok = 35 <= df['rsi'].iloc[i] <= 50
                macd_bullish = df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]
                
                if price_near_ema and rsi_ok and macd_bullish:
                    # ML confirmation
                    features_vector = scaler.transform([df[features].iloc[i].fillna(0)])
                    ml_prediction = ml_model.predict_proba(features_vector)[0][1]
                    
                    if ml_prediction > 0.7:  # 70% confidence threshold
                        df.loc[df.index[i], 'signal'] = 1
            
            elif trend_down and adx_strong:
                # Short conditions
                price_near_ema = abs(df['close'].iloc[i] - df['ema_50'].iloc[i]) / df['ema_50'].iloc[i] < 0.005
                rsi_ok = 50 <= df['rsi'].iloc[i] <= 65
                macd_bearish = df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]
                
                if price_near_ema and rsi_ok and macd_bearish:
                    # ML confirmation
                    features_vector = scaler.transform([df[features].iloc[i].fillna(0)])
                    ml_prediction = ml_model.predict_proba(features_vector)[0][1]
                    
                    if ml_prediction > 0.7:
                        df.loc[df.index[i], 'signal'] = -1
        
        # Calculate exits with trailing stop
        df['position'] = df['signal']
        df['stop_loss'] = 0.0
        df['trailing_stop'] = 0.0
        
        # Simulate positions with risk management
        initial_capital = 10000
        position = 0
        entry_price = 0
        stop_loss = 0
        trailing_stop = 0
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1 and position == 0:  # Long entry
                position = 1
                entry_price = df['close'].iloc[i]
                atr = df['atr'].iloc[i]
                stop_loss = entry_price - (atr * 2.2)
                trailing_stop = stop_loss
                
            elif df['signal'].iloc[i] == -1 and position == 0:  # Short entry
                position = -1
                entry_price = df['close'].iloc[i]
                atr = df['atr'].iloc[i]
                stop_loss = entry_price + (atr * 2.2)
                trailing_stop = stop_loss
                
            elif position != 0:
                # Update trailing stop
                if position == 1:  # Long
                    new_trailing = df['close'].iloc[i] - (df['atr'].iloc[i] * 2.0)
                    trailing_stop = max(trailing_stop, new_trailing)
                    if df['low'].iloc[i] <= trailing_stop:
                        position = 0
                        
                elif position == -1:  # Short
                    new_trailing = df['close'].iloc[i] + (df['atr'].iloc[i] * 2.0)
                    trailing_stop = min(trailing_stop, new_trailing)
                    if df['high'].iloc[i] >= trailing_stop:
                        position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        # Calculate returns
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Calculate metrics
        total_return = df['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = self.calculate_sharpe(df['strategy_returns'])
        max_drawdown = self.calculate_max_drawdown(df['cumulative_returns'])
        win_rate = self.calculate_win_rate(df['strategy_returns'])
        
        self.results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'signals_df': df
        }
        
        return self.results
    
    def calculate_sharpe(self, returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_max_drawdown(self, cumulative_returns):
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def calculate_win_rate(self, returns):
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        return winning_trades / total_trades if total_trades > 0 else 0
    
    def generate_report(self):
        """Generate detailed backtest report"""
        results = self.results
        
        print("=" * 60)
        print("ETH/USD STRATEGY BACKTEST REPORT")
        print("=" * 60)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {self.calculate_profit_factor():.2f}")
        print(f"Average Win: {self.calculate_avg_win():.2%}")
        print(f"Average Loss: {self.calculate_avg_loss():.2%}")
        print(f"Total Trades: {self.count_trades()}")
        print("=" * 60)
        
        # Plot equity curve
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(results['signals_df']['cumulative_returns'])
        plt.title('Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.show()
