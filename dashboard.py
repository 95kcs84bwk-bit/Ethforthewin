import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import threading
import time

class TradingDashboard:
    def __init__(self, bot):
        self.bot = bot
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("ETH Trading Bot Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Account Overview"),
                    html.P(id='balance-display'),
                    html.P(id='daily-pnl-display'),
                    html.P(id='open-positions-display'),
                    html.P(id='win-rate-display')
                ], className='four columns'),
                
                html.Div([
                    html.H3("Recent Signals"),
                    html.Table(id='signals-table')
                ], className='eight columns'),
            ], className='row'),
            
            dcc.Graph(id='price-chart'),
            dcc.Graph(id='equity-curve'),
            
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # Update every 10 seconds
                n_intervals=0
            )
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('balance-display', 'children'),
             Output('daily-pnl-display', 'children'),
             Output('open-positions-display', 'children'),
             Output('win-rate-display', 'children'),
             Output('signals-table', 'children'),
             Output('price-chart', 'figure'),
             Output('equity-curve', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get current data from bot
            balance = self.bot.get_account_balance()
            daily_pnl = self.bot.daily_pnl
            open_positions = len(self.bot.positions)
            
            # Calculate win rate
            if len(self.bot.pnl_history) > 0:
                wins = sum(1 for pnl in self.bot.pnl_history if pnl > 0)
                win_rate = wins / len(self.bot.pnl_history) * 100
            else:
                win_rate = 0
            
            # Create signals table
            signals_data = []
            if self.bot.last_signal:
                signal = self.bot.last_signal
                signals_data.append([
                    signal.timestamp.strftime('%Y-%m-%d %H:%M'),
                    signal.side.value,
                    f"${signal.entry_price:.2f}",
                    f"${signal.stop_loss:.2f}",
                    f"{signal.confidence:.1%}"
                ])
            
            signals_table = html.Table([
                html.Thead(html.Tr([
                    html.Th('Time'), html.Th('Side'), 
                    html.Th('Entry'), html.Th('Stop'), html.Th('Confidence')
                ])),
                html.Tbody([
                    html.Tr([html.Td(cell) for cell in row]) 
                    for row in signals_data
                ])
            ])
            
            # Create price chart
            price_fig = go.Figure()
            # Add price data here from bot
            
            # Create equity curve
            equity_fig = go.Figure()
            # Add equity data here
            
            return [
                f"Balance: ${balance:.2f}",
                f"Daily PnL: ${daily_pnl:.2f}",
                f"Open Positions: {open_positions}",
                f"Win Rate: {win_rate:.1f}%",
                signals_table,
                price_fig,
                equity_fig
            ]
    
    def run(self, port=8050):
        self.app.run_server(debug=False, port=port)
