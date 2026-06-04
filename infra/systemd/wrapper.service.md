
wrapper-service placed under `$HOME/.config/systemd/user`

```
# /etc/systemd/system/llama-wrapper.service
[Unit]
Description=Generic wrapper for llama.cpp
After=network.target
Wants=network.target

[Service]
# NoNewPrivileges=false
# CapabilityBoundingSet=CAP_IPC_LOCK
# AmbientCapabilities=CAP_IPC_LOCK
Type=simple
# For --user , User and Group cannot be used
# User=llama
# Group=llama,video,render,agents
WorkingDirectory=/home/llama/aibin
ExecStart=/bin/bash -c 'cd "$WORKING_DIR" && exec /home/llama/aibin/llama_start.sh "$@"'
Environment=PATH=/home/llama/.local/bin:/home/llama/aibin:/usr/local/bin:/usr/bin:/bin

# Restart policy: Restart if crashed
Restart=on-failure
RestartSec=10
TimeoutStopSec=15

# Logging
StandardOutput=journal
StandardError=journal

# Security
SyslogIdentifier=llama-wrapper
EnvironmentFile=/home/llama/.local/share/llama-config/llama-env.sh

[Install]
WantedBy=default.target
```

**some aliases**
```shell
alias jcfollow='journalctl --user -u llama-wrapper -f'
alias jcfollow2='journalctl --user -u llama-wrapper-2 -f'
alias jcreverse='journalctl --user -u llama-wrapper -r'
alias jcreverse2='journalctl --user -u llama-wrapper-2 -r'
alias scdisable='systemctl --user enable llama-wrapper.service'
alias scdisable2='systemctl --user enable llama-wrapper-2.service'
alias scenable='systemctl --user enable llama-wrapper.service'
alias scenable2='systemctl --user enable llama-wrapper-2.service'
alias screstart='systemctl --user restart llama-wrapper.service'
alias screstart2='systemctl --user restart llama-wrapper-2.service'
alias scstart='systemctl --user start llama-wrapper.service'
alias scstart2='systemctl --user start llama-wrapper-2.service'
alias scstatus='systemctl --user status llama-wrapper.service'
alias scstatus2='systemctl --user status llama-wrapper-2.service'
alias scstop='systemctl --user stop llama-wrapper.service'
alias scstop2='systemctl --user stop llama-wrapper-2.service'

```