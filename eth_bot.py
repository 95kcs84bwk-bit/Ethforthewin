import pandas as pd
import numpy as np
import talib as ta
import ccxt
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Tuple, Optional
import threading
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eth_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

@dataclass
class TradeSignal:
    side: TradeSide
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    timestamp: datetime
    position_size: float

class ETHTradingBot:
    def __init__(self, config: Dict):
        self.config = config
        self.exchange = self._init_exchange()
        self.positions = {}
        self.pnl_history = []
        self.last_signal = None
        
        # Initialize indicators cache
        self.indicators = {
            'm30': {}, 'h4': {}, 'd1': {}
        }
        
        # Risk management
        self.daily_pnl = 0
        self.daily_trades = 0
        self.consecutive_losses = 0
        
    def _init_exchange(self):
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt, self.config['exchange'])
        return exchange_class({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} if self.config['use_futures'] else {}
        })
    
    def fetch_ohlcv(self, timeframe: str, limit: int = 500):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.config['symbol'],
                timeframe=timeframe,
                limit=limit
            )
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame, timeframe: str):
        """Calculate all required indicators"""
        if df is None or len(df) < 200:
            return None
            
        # EMAs
        df['ema_21'] = ta.EMA(df['close'], timeperiod=21)
        df['ema_55'] = ta.EMA(df['close'], timeperiod=55)
        df['ema_200'] = ta.EMA(df['close'], timeperiod=200)
        df['ema_50'] = ta.EMA(df['close'], timeperiod=50)
        
        # RSI with custom bands
        df['rsi'] = ta.RSI(df['close'], timeperiod=11)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
            df['close'], 
            fastperiod=8, 
            slowperiod=21, 
            signalperiod=5
        )
        
        # ADX for trend strength
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ATR for volatility and stops
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
            df['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2
        )
        
        # Volume indicators
        df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def analyze_trend(self, h4_df: pd.DataFrame) -> Tuple[TradeSide, float]:
        """Analyze H4 trend for filter"""
        if h4_df is None or len(h4_df) < 50:
            return TradeSide.NEUTRAL, 0.0
            
        last = h4_df.iloc[-1]
        
        # Trend direction
        trend_up = last['close'] > last['ema_200']
        trend_strength = last['adx']
        
        if trend_strength < 25:
            return TradeSide.NEUTRAL, trend_strength
            
        return TradeSide.LONG if trend_up else TradeSide.SHORT, trend_strength
    
    def generate_signal(self, m30_df: pd.DataFrame, trend: TradeSide, trend_strength: float) -> Optional[TradeSignal]:
        """Generate trading signal based on M30 data"""
        if m30_df is None or len(m30_df) < 50:
            return None
            
        last = m30_df.iloc[-1]
        prev = m30_df.iloc[-2]
        
        # Check entry conditions based on trend
        if trend == TradeSide.LONG:
            # Long entry conditions
            price_near_ema = abs(last['close'] - last['ema_50']) / last['ema_50'] < 0.005
            rsi_condition = 35 <= last['rsi'] <= 50
            macd_bullish = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
            volume_spike = last['volume_ratio'] > 1.5
            
            if price_near_ema and rsi_condition and (macd_bullish or volume_spike):
                # Calculate position size
                atr = last['atr']
                stop_loss = last['low'] - (atr * 2.2)
                risk = last['close'] - stop_loss
                position_size = self.calculate_position_size(risk)
                
                # Calculate take profits
                tp1 = last['close'] + (risk * 1.5)
                tp2 = last['close'] + (risk * 3.0)
                
                confidence = min(0.95, trend_strength / 100 + 0.5)
                
                return TradeSignal(
                    side=TradeSide.LONG,
                    entry_price=last['close'],
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    position_size=position_size
                )
                
        elif trend == TradeSide.SHORT:
            # Short entry conditions
            price_near_ema = abs(last['close'] - last['ema_50']) / last['ema_50'] < 0.005
            rsi_condition = 50 <= last['rsi'] <= 65
            macd_bearish = last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']
            volume_spike = last['volume_ratio'] > 1.5
            
            if price_near_ema and rsi_condition and (macd_bearish or volume_spike):
                # Calculate position size
                atr = last['atr']
                stop_loss = last['high'] + (atr * 2.2)
                risk = stop_loss - last['close']
                position_size = self.calculate_position_size(risk)
                
                # Calculate take profits
                tp1 = last['close'] - (risk * 1.5)
                tp2 = last['close'] - (risk * 3.0)
                
                confidence = min(0.95, trend_strength / 100 + 0.5)
                
                return TradeSignal(
                    side=TradeSide.SHORT,
                    entry_price=last['close'],
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    position_size=position_size
                )
        
        return None
    
    def calculate_position_size(self, risk_per_unit: float) -> float:
        """Calculate position size based on risk management rules"""
        account_balance = self.get_account_balance()
        risk_per_trade = account_balance * self.config['risk_per_trade']
        
        # Adjust for consecutive losses
        if self.consecutive_losses > 2:
            risk_per_trade *= 0.5
        elif self.consecutive_losses > 0:
            risk_per_trade *= 0.75
        
        # Check daily drawdown
        if abs(self.daily_pnl) > account_balance * self.config['max_daily_drawdown']:
            return 0
        
        position_size = risk_per_trade / risk_per_unit
        max_position = account_balance * 0.1 / self.get_current_price()  # Max 10% of account
        
        return min(position_size, max_position)
    
    def execute_trade(self, signal: TradeSignal):
        """Execute the trade based on signal"""
        if not self.check_trading_conditions():
            logger.warning("Trading conditions not met, skipping trade")
            return
            
        try:
            # Place initial order
            order = self.exchange.create_order(
                symbol=self.config['symbol'],
                type='limit',
                side='buy' if signal.side == TradeSide.LONG else 'sell',
                amount=signal.position_size,
                price=signal.entry_price
            )
            
            # Set stop loss and take profit orders
            self.place_stop_loss(order['id'], signal)
            self.place_take_profits(order['id'], signal)
            
            logger.info(f"Trade executed: {signal}")
            
            # Track position
            self.positions[order['id']] = {
                'signal': signal,
                'entry_time': datetime.now(),
                'initial_size': signal.position_size,
                'remaining_size': signal.position_size
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    def place_stop_loss(self, order_id: str, signal: TradeSignal):
        """Place stop loss order"""
        try:
            sl_order = self.exchange.create_order(
                symbol=self.config['symbol'],
                type='stop_market',
                side='sell' if signal.side == TradeSide.LONG else 'buy',
                amount=signal.position_size,
                params={
                    'stopPrice': signal.stop_loss,
                    'reduceOnly': True
                }
            )
            logger.info(f"Stop loss placed at {signal.stop_loss}")
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
    
    def place_take_profits(self, order_id: str, signal: TradeSignal):
        """Place take profit orders"""
        try:
            # TP1 for 50% of position
            tp1_size = signal.position_size * 0.5
            tp1_order = self.exchange.create_order(
                symbol=self.config['symbol'],
                type='limit',
                side='sell' if signal.side == TradeSide.LONG else 'buy',
                amount=tp1_size,
                price=signal.take_profit_1,
                params={'reduceOnly': True}
            )
            
            # TP2 for 25% of position
            tp2_size = signal.position_size * 0.25
            tp2_order = self.exchange.create_order(
                symbol=self.config['symbol'],
                type='limit',
                side='sell' if signal.side == TradeSide.LONG else 'buy',
                amount=tp2_size,
                price=signal.take_profit_2,
                params={'reduceOnly': True}
            )
            
            logger.info(f"Take profits placed at {signal.take_profit_1} and {signal.take_profit_2}")
            
        except Exception as e:
            logger.error(f"Failed to place take profits: {e}")
    
    def check_trading_conditions(self) -> bool:
        """Check if trading conditions are met"""
        # Check time (focus on high liquidity hours)
        now_utc = datetime.utcnow()
        if not (13 <= now_utc.hour <= 21):
            logger.info("Outside optimal trading hours")
            return False
        
        # Check volatility
        volatility = self.calculate_volatility()
        if volatility < 0.015:  # 1.5% minimum volatility
            logger.info(f"Volatility too low: {volatility:.2%}")
            return False
        
        # Check daily trade limit
        if self.daily_trades >= self.config['max_daily_trades']:
            logger.info("Daily trade limit reached")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            logger.warning("Max consecutive losses reached, pausing")
            return False
        
        return True
    
    def calculate_volatility(self) -> float:
        """Calculate 24h volatility"""
        try:
            ticker = self.exchange.fetch_ticker(self.config['symbol'])
            high_24h = ticker['high']
            low_24h = ticker['low']
            return (high_24h - low_24h) / low_24h
        except:
            return 0.02  # Default value
    
    def get_account_balance(self) -> float:
        """Get available account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free'] if self.config['use_futures'] else balance['total']['USDT']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self.config['initial_balance']
    
    def get_current_price(self) -> float:
        """Get current ETH price"""
        try:
            ticker = self.exchange.fetch_ticker(self.config['symbol'])
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0
    
    def run(self):
        """Main bot execution loop"""
        logger.info("Starting ETH Trading Bot...")
        
        while True:
            try:
                # Fetch data for different timeframes
                m30_data = self.fetch_ohlcv('30m')
                h4_data = self.fetch_ohlcv('4h')
                
                if m30_data is None or h4_data is None:
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                m30_df = self.calculate_indicators(m30_data, '30m')
                h4_df = self.calculate_indicators(h4_data, '4h')
                
                # Analyze trend
                trend, trend_strength = self.analyze_trend(h4_df)
                
                # Generate signal
                signal = self.generate_signal(m30_df, trend, trend_strength)
                
                if signal and signal.confidence > self.config['min_confidence']:
                    # Check if we already have a similar position
                    if not self.has_similar_position(signal):
                        self.execute_trade(signal)
                        self.last_signal = signal
                
                # Monitor open positions
                self.monitor_positions()
                
                # Log status
                self.log_status()
                
                # Sleep until next candle (adjust for real-time)
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def has_similar_position(self, signal: TradeSignal) -> bool:
        """Check if we already have a similar position"""
        for pos_id, position in self.positions.items():
            if position['signal'].side == signal.side:
                # Check if price is within 2% of existing position
                price_diff = abs(position['signal'].entry_price - signal.entry_price) / signal.entry_price
                if price_diff < 0.02:
                    return True
        return False
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        current_price = self.get_current_price()
        
        for pos_id, position in list(self.positions.items()):
            signal = position['signal']
            pnl_pct = self.calculate_pnl(position, current_price)
            
            # Check for trailing stop on remaining 25%
            if position['remaining_size'] > 0:
                self.update_trailing_stop(pos_id, signal, current_price)
            
            # Check if stop loss or take profits were hit
            # (In production, this would check order status via exchange API)
            
            # Log position status
            logger.debug(f"Position {pos_id}: PnL {pnl_pct:.2%}")
    
    def calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate PnL for a position"""
        signal = position['signal']
        if signal.side == TradeSide.LONG:
            return (current_price - signal.entry_price) / signal.entry_price
        else:
            return (signal.entry_price - current_price) / signal.entry_price
    
    def update_trailing_stop(self, pos_id: str, signal: TradeSignal, current_price: float):
        """Update trailing stop for remaining position"""
        # This would update the stop loss order on exchange
        # Implementation depends on exchange API
        pass
    
    def log_status(self):
        """Log current bot status"""
        balance = self.get_account_balance()
        open_positions = len(self.positions)
        
        logger.info(f"Status - Balance: ${balance:.2f}, Positions: {open_positions}, "
                   f"Daily PnL: ${self.daily_pnl:.2f}, Daily Trades: {self.daily_trades}")
