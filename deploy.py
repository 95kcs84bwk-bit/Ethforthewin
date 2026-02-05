import sys
import subprocess
import os
from datetime import datetime

class BotDeployer:
    @staticmethod
    def deploy_on_vps():
        """Deploy bot on VPS with PM2 for process management"""
        deploy_script = """
        #!/bin/bash
        # Update system
        sudo apt-get update
        sudo apt-get upgrade -y
        
        # Install Python and dependencies
        sudo apt-get install -y python3-pip python3-venv git
        
        # Clone bot repository
        git clone https://github.com/your-repo/eth-trading-bot.git
        cd eth-trading-bot
        
        # Setup virtual environment
        python3 -m venv bot_env
        source bot_env/bin/activate
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Install PM2 for process management
        npm install pm2 -g
        
        # Create PM2 ecosystem file
        cat > ecosystem.config.js << 'EOF'
        module.exports = {
          apps: [{
            name: 'eth-bot',
            script: 'eth_bot.py',
            interpreter: 'bot_env/bin/python',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '1G',
            env: {
              NODE_ENV: 'production'
            },
            error_file: 'logs/err.log',
            out_file: 'logs/out.log',
            log_file: 'logs/combined.log',
            time: true
          }]
        };
        EOF
        
        # Start bot with PM2
        pm2 start ecosystem.config.js
        pm2 save
        pm2 startup
        
        echo "Bot deployed successfully!"
        """
        
        print("Run this script on your VPS:")
        print(deploy_script)
    
    @staticmethod
    def create_docker_compose():
        """Create Docker deployment configuration"""
        docker_compose = """
        version: '3.8'
        services:
          eth-bot:
            build: .
            container_name: eth-trading-bot
            restart: unless-stopped
            volumes:
              - ./config:/app/config
              - ./logs:/app/logs
              - ./data:/app/data
            environment:
              - TZ=UTC
            logging:
              driver: "json-file"
              options:
                max-size: "10m"
                max-file: "3"
        
          monitoring:
            image: grafana/grafana
            ports:
              - "3000:3000"
            volumes:
              - grafana-storage:/var/lib/grafana
            restart: unless-stopped
        
        volumes:
          grafana-storage:
        """
        
        with open('docker-compose.yml', 'w') as f:
            f.write(docker_compose)
        
        print("Docker compose file created!")
