#!/bin/bash

# Create systemd service file for Passivbot distributed optimization GPU client
cat > passivbot-gpu-client.service << EOF
[Unit]
Description=Passivbot Distributed Optimization GPU Client
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(which python3) src/distributed_optimize_gpu.py --mode client --server SERVER_ADDRESS --max-cpu 70 --max-memory 80 --max-gpu 80 --workers 0 --use-gpu
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=multi-user.target
EOF

# Replace SERVER_ADDRESS with actual server address
read -p "Enter server address (e.g., 192.168.1.100:5555): " server_address
sed -i "s/SERVER_ADDRESS/$server_address/g" passivbot-gpu-client.service

# Install the service
sudo mv passivbot-gpu-client.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable passivbot-gpu-client.service

echo "Passivbot GPU client service installed and enabled."
echo "To start the service: sudo systemctl start passivbot-gpu-client.service"
echo "To check status: sudo systemctl status passivbot-gpu-client.service"
echo "To stop the service: sudo systemctl stop passivbot-gpu-client.service"