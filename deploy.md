# AI-Enhanced Passivbot + PBGUI Deployment Guide

This guide provides complete deployment instructions for setting up the AI-enhanced Passivbot system and PBGUI web interface on your remote server.

## Prerequisites

- Remote server with GPU support (NVIDIA recommended)
- SSH access to server
- Python 3.8+ installed
- Git installed

## Server Specifications

**Target Server:** root@192.168.55.250
- **GPU:** NVIDIA GeForce GTX 1080 Ti (11.7 GB CUDA memory)
- **OS:** Linux (Ubuntu/Debian recommended)
- **CUDA Support:** Version 6.1+ (compatible with older GPUs)

## 1. Initial Server Setup

### 1.1 Connect to Server
```bash
ssh root@192.168.55.250
```

### 1.2 Update System
```bash
apt update && apt upgrade -y
apt install -y git python3 python3-pip python3-venv build-essential
```

### 1.3 Install CUDA Dependencies (if not already installed)
```bash
# Check CUDA installation
nvidia-smi

# If CUDA not installed, follow NVIDIA's installation guide for your OS
# https://developer.nvidia.com/cuda-downloads
```

## 2. Clone and Setup Passivbot

### 2.1 Create Project Directory
```bash
cd /root
mkdir -p ai-trading-server
cd ai-trading-server
```

### 2.2 Clone AI-Enhanced Passivbot
```bash
git clone https://github.com/enarjord/passivbot.git
cd passivbot

# Note: Apply AI enhancements from your local development
# Copy your enhanced files to this directory
```

### 2.3 Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.4 Install Dependencies
```bash
# Install Rust (required for passivbot-rust)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build Rust components
cd passivbot-rust
cargo build --release
cd ..
```

### 2.5 Install AI Dependencies
```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional AI dependencies
pip install torchrl>=0.3.0 gymnasium>=0.28.0 tensorboard>=2.13.0
```

### 2.6 Copy Enhanced AI Files
Copy your enhanced AI files from local development:

```bash
# From your local machine, copy the AI optimizer files
scp -r /Users/Agentspades-work/Documents/crypto/ai-passivbot/src/ai_optimizer/ root@192.168.55.250:/root/ai-trading-server/passivbot/src/
scp /Users/Agentspades-work/Documents/crypto/ai-passivbot/src/optimize.py root@192.168.55.250:/root/ai-trading-server/passivbot/src/
scp /Users/Agentspades-work/Documents/crypto/ai-passivbot/test_ai_optimizer.py root@192.168.55.250:/root/ai-trading-server/passivbot/
scp /Users/Agentspades-work/Documents/crypto/ai-passivbot/src/auto_optimizer.py root@192.168.55.250:/root/ai-trading-server/passivbot/src/
scp -r /Users/Agentspades-work/Documents/crypto/ai-passivbot/docs/ root@192.168.55.250:/root/ai-trading-server/passivbot/
```

## 3. Setup PBGUI

### 3.1 Clone PBGUI
```bash
cd /root/ai-trading-server
git clone https://github.com/msei99/pbgui.git
cd pbgui
```

### 3.2 Install PBGUI Dependencies
```bash
python3 -m venv pbgui-venv
source pbgui-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit
```

### 3.3 Copy AI-Enhanced PBGUI Files
```bash
# From your local machine, copy enhanced PBGUI files
scp /Users/Agentspades-work/Documents/crypto/pbgui/AIOptimize.py root@192.168.55.250:/root/ai-trading-server/pbgui/
scp /Users/Agentspades-work/Documents/crypto/pbgui/AIDashboard.py root@192.168.55.250:/root/ai-trading-server/pbgui/
scp /Users/Agentspades-work/Documents/crypto/pbgui/AIConfig.py root@192.168.55.250:/root/ai-trading-server/pbgui/
```

## 4. Configuration

### 4.1 Configure Passivbot
```bash
cd /root/ai-trading-server/passivbot

# Copy and customize configuration
cp api-keys.json.example api-keys.json
# Edit api-keys.json with your exchange API credentials

# Test basic passivbot functionality
python3 src/optimize.py --help
```

### 4.2 Test AI System
```bash
cd /root/ai-trading-server/passivbot
source venv/bin/activate

# Run AI system tests
python test_ai_optimizer.py

# Should show: "🎉 All tests passed! AI optimization system is ready to use."
```

### 4.3 Configure PBGUI
```bash
cd /root/ai-trading-server/pbgui

# Configure PBGUI settings
# Edit pbgui.ini with your preferences
```

## 5. Service Setup (Systemd)

### 5.1 Create PBGUI Service
```bash
sudo tee /etc/systemd/system/pbgui.service > /dev/null << EOF
[Unit]
Description=PBGUI Web Interface
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai-trading-server/pbgui
Environment=PATH=/root/ai-trading-server/pbgui/pbgui-venv/bin
ExecStart=/root/ai-trading-server/pbgui/pbgui-venv/bin/python -m streamlit run pbgui.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

### 5.2 Create Auto-Optimizer Service (Optional)
```bash
sudo tee /etc/systemd/system/passivbot-ai.service > /dev/null << EOF
[Unit]
Description=Passivbot AI Auto-Optimizer
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai-trading-server/passivbot
Environment=PATH=/root/ai-trading-server/passivbot/venv/bin
ExecStart=/root/ai-trading-server/passivbot/venv/bin/python src/auto_optimizer.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
```

### 5.3 Enable and Start Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable pbgui
sudo systemctl start pbgui

# Optional: Enable AI auto-optimizer
# sudo systemctl enable passivbot-ai
# sudo systemctl start passivbot-ai
```

## 6. Firewall Configuration

### 6.1 Open Required Ports
```bash
# Open PBGUI port (8501)
sudo ufw allow 8501/tcp

# Open SSH if not already open
sudo ufw allow ssh

# Enable firewall
sudo ufw --force enable
```

## 7. Access and Testing

### 7.1 Access PBGUI
Open your web browser and navigate to:
```
http://192.168.55.250:8501
```

### 7.2 Test AI Features
1. Navigate to the AI Optimization section in PBGUI
2. Configure AI training parameters
3. Start AI optimization:
   ```bash
   cd /root/ai-trading-server/passivbot
   source venv/bin/activate
   python src/optimize.py --ai-mode --ai-episodes 100
   ```

### 7.3 Monitor Services
```bash
# Check service status
sudo systemctl status pbgui
sudo systemctl status passivbot-ai  # if enabled

# View logs
sudo journalctl -u pbgui -f
sudo journalctl -u passivbot-ai -f  # if enabled
```

## 8. Maintenance and Updates

### 8.1 Update Passivbot
```bash
cd /root/ai-trading-server/passivbot
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
cd passivbot-rust && cargo build --release && cd ..
```

### 8.2 Update PBGUI
```bash
cd /root/ai-trading-server/pbgui
git pull origin main
source pbgui-venv/bin/activate
pip install -r requirements.txt
```

### 8.3 Restart Services
```bash
sudo systemctl restart pbgui
sudo systemctl restart passivbot-ai  # if enabled
```

## 9. Troubleshooting

### 9.1 Common Issues

**GPU Not Detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Import Errors:**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Port Already in Use:**
```bash
# Kill process using port 8501
sudo lsof -ti:8501 | xargs sudo kill -9

# Or change port in PBGUI service file
```

### 9.2 Log Locations
- PBGUI logs: `sudo journalctl -u pbgui`
- Passivbot AI logs: `sudo journalctl -u passivbot-ai`
- System logs: `/var/log/syslog`

### 9.3 Performance Monitoring
```bash
# Monitor GPU usage
watch nvidia-smi

# Monitor system resources
htop

# Monitor disk usage
df -h
```

## 10. Security Considerations

### 10.1 API Key Security
- Store API keys in secure files with restricted permissions
- Use environment variables for sensitive configuration
- Never commit API keys to version control

### 10.2 Network Security
- Use VPN for remote access when possible
- Configure fail2ban for SSH protection
- Regularly update system packages

### 10.3 Backup Strategy
```bash
# Backup configurations
tar -czf passivbot-backup-$(date +%Y%m%d).tar.gz /root/ai-trading-server/

# Store backups securely offsite
```

## 11. Usage Examples

### 11.1 AI-Only Optimization
```bash
cd /root/ai-trading-server/passivbot
source venv/bin/activate
python src/optimize.py --ai-mode --ai-episodes 500 --symbol BTCUSDT
```

### 11.2 Hybrid Optimization
```bash
python src/optimize.py --hybrid-mode --ai-episodes 200 --population-size 100
```

### 11.3 Auto-Optimization (Live Trading)
```bash
python src/auto_optimizer.py --config configs/live_config.json
```

---

## Support

For issues and questions:
- Check logs using systemctl and journalctl
- Review AI architecture documentation in `/docs/`
- Test individual components before full deployment
- Monitor GPU memory usage during training

**Deployment Complete!** Your AI-enhanced Passivbot system with PBGUI is now ready for production trading.