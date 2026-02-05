class RiskMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.alert_thresholds = {
            'daily_drawdown': 0.02,
            'position_concentration': 0.3,
            'correlation_risk': 0.8,
            'volatility_spike': 3.0
        }
        self.alerts = []
    
    def run_checks(self):
        """Run all risk checks"""
        checks = [
            self.check_daily_drawdown,
            self.check_position_concentration,
            self.check_correlation,
            self.check_volatility,
            self.check_liquidity,
            self.check_market_regime
        ]
        
        for check in checks:
            result = check()
            if not result['pass']:
                self.alerts.append(result['message'])
                self.take_action(result['severity'])
    
    def check_daily_drawdown(self):
        """Check if daily drawdown exceeds threshold"""
        drawdown = abs(self.bot.daily_pnl) / self.bot.get_account_balance()
        
        if drawdown > self.alert_thresholds['daily_drawdown']:
            return {
                'pass': False,
                'severity': 'HIGH',
                'message': f'Daily drawdown {drawdown:.2%} exceeds threshold'
            }
        return {'pass': True}
    
    def check_position_concentration(self):
        """Check if position concentration is too high"""
        total_exposure = sum(
            pos['signal'].position_size * pos['signal'].entry_price
            for pos in self.bot.positions.values()
        )
        
        concentration = total_exposure / self.bot.get_account_balance()
        
        if concentration > self.alert_thresholds['position_concentration']:
            return {
                'pass': False,
                'severity': 'MEDIUM',
                'message': f'Position concentration {concentration:.2%} too high'
            }
        return {'pass': True}
    
    def take_action(self, severity):
        """Take action based on risk severity"""
        if severity == 'HIGH':
            # Close all positions
            self.bot.close_all_positions()
            logger.critical("HIGH RISK - All positions closed")
        
        elif severity == 'MEDIUM':
            # Reduce position sizes
            self.bot.reduce_position_sizes(0.5)
            logger.warning("MEDIUM RISK - Position sizes reduced by 50%")
        
        elif severity == 'LOW':
            # Pause new entries
            self.bot.pause_trading()
            logger.warning("LOW RISK - Trading paused temporarily")
