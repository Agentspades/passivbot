#!/bin/bash

# Passivbot Distributed Optimization Server Setup Script for Raspberry Pi
# This script sets up a Passivbot distributed optimization server as a systemd service

# Exit on any error
set -e

# Configuration variables - modify these as needed
PASSIVBOT_USER="pi"
PASSIVBOT_HOME="/home/$PASSIVBOT_USER/passivbot"  # Path to your existing repo
SERVER_PORT="5555"
CONFIG_PATH="configs/100-scalping-forager.json"
BATCH_SIZE="32"
LOG_LEVEL="info"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root or with sudo${NC}"
  exit 1
fi

echo -e "${GREEN}=== Passivbot Distributed Optimization Server Setup for Raspberry Pi ===${NC}"
echo -e "${YELLOW}This script will set up a Passivbot distributed optimization server as a systemd service.${NC}"
echo ""

# Update system packages
echo -e "${GREEN}Updating system packages...${NC}"
apt-get update
apt-get upgrade -y

# Install required dependencies
echo -e "${GREEN}Installing required dependencies...${NC}"
apt-get install -y python3 python3-pip python3-venv git ufw

# Verify the repository exists
if [ ! -d "$PASSIVBOT_HOME" ]; then
    echo -e "${RED}Error: Passivbot repository not found at $PASSIVBOT_HOME${NC}"
    echo -e "${YELLOW}Please clone the repository first or update PASSIVBOT_HOME in the script.${NC}"
    exit 1
else
    echo -e "${GREEN}Found Passivbot repository at $PASSIVBOT_HOME${NC}"
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "$PASSIVBOT_HOME/venv" ]; then
    echo -e "${GREEN}Setting up Python virtual environment...${NC}"
    su - "$PASSIVBOT_USER" -c "cd $PASSIVBOT_HOME && python3 -m venv venv"
else
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
fi

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
su - "$PASSIVBOT_USER" -c "cd $PASSIVBOT_HOME && source venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install pyzmq numpy psutil tqdm"

# Create logs directory
echo -e "${GREEN}Creating logs directory...${NC}"
su - "$PASSIVBOT_USER" -c "mkdir -p $PASSIVBOT_HOME/logs"

# Create systemd service file
echo -e "${GREEN}Creating systemd service...${NC}"
cat > /etc/systemd/system/passivbot-server.service << EOF
[Unit]
Description=Passivbot Distributed Optimization Server
After=network.target

[Service]
Type=simple
User=$PASSIVBOT_USER
WorkingDirectory=$PASSIVBOT_HOME
ExecStart=$PASSIVBOT_HOME/venv/bin/python $PASSIVBOT_HOME/src/distributed_optimize_v3.py --mode server --config $CONFIG_PATH --port $SERVER_PORT --batch-size $BATCH_SIZE --log-level $LOG_LEVEL
Restart=on-failure
RestartSec=10
StandardOutput=append:$PASSIVBOT_HOME/logs/server.log
StandardError=append:$PASSIVBOT_HOME/logs/server-error.log
Environment="PATH=$PASSIVBOT_HOME/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# Create a convenience script to check logs
echo -e "${GREEN}Creating log viewing script...${NC}"
cat > /usr/local/bin/passivbot-logs << EOF
#!/bin/bash
tail -f $PASSIVBOT_HOME/logs/server.log
EOF
chmod +x /usr/local/bin/passivbot-logs

# Create a convenience script to check error logs
cat > /usr/local/bin/passivbot-errors << EOF
#!/bin/bash
tail -f $PASSIVBOT_HOME/logs/server-error.log
EOF
chmod +x /usr/local/bin/passivbot-errors

# Create a convenience script to manage the service
echo -e "${GREEN}Creating service management script...${NC}"
cat > /usr/local/bin/passivbot-service << EOF
#!/bin/bash
case "\$1" in
  start)
    systemctl start passivbot-server
    echo "Passivbot server started"
    ;;
  stop)
    systemctl stop passivbot-server
    echo "Passivbot server stopped"
    ;;
  restart)
    systemctl restart passivbot-server
    echo "Passivbot server restarted"
    ;;
  status)
    systemctl status passivbot-server
    ;;
  *)
    echo "Usage: passivbot-service {start|stop|restart|status}"
    exit 1
    ;;
esac
EOF
chmod +x /usr/local/bin/passivbot-service

# Configure firewall
echo -e "${GREEN}Configuring firewall...${NC}"
ufw allow ssh
ufw allow "$SERVER_PORT"/tcp
ufw allow "$((SERVER_PORT + 1))"/tcp  # For ZMQ publisher port
ufw --force enable

# Reload systemd, enable and start service
echo -e "${GREEN}Enabling and starting service...${NC}"
systemctl daemon-reload
systemctl enable passivbot-server
systemctl start passivbot-server

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
systemctl status passivbot-server --no-pager

# Get the Raspberry Pi's IP address
PI_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo -e "${YELLOW}The Passivbot distributed optimization server is now running as a service.${NC}"
echo ""
echo -e "You can manage the service with: ${GREEN}passivbot-service {start|stop|restart|status}${NC}"
echo -e "View logs with: ${GREEN}passivbot-logs${NC}"
echo -e "View error logs with: ${GREEN}passivbot-errors${NC}"
echo ""
echo -e "Server is listening on: ${GREEN}$PI_IP:$SERVER_PORT${NC}"
echo -e "Configuration file: ${GREEN}$CONFIG_PATH${NC}"
echo ""
echo -e "${YELLOW}To connect clients, use:${NC}"
echo -e "${GREEN}python src/distributed_optimize_v3.py --mode client --server $PI_IP:$SERVER_PORT${NC}"
echo ""
echo -e "${YELLOW}Note: For Raspberry Pi, consider these performance tips:${NC}"
echo -e "1. Use a cooling solution to prevent thermal throttling"
echo -e "2. Consider overclocking if you have adequate cooling"
echo -e "3. Use a high-quality SD card or USB SSD for better I/O performance"
echo -e "4. Adjust batch-size to a smaller value (8-16) to reduce memory pressure"
echo -e "5. Monitor system temperature with 'vcgencmd measure_temp'"
