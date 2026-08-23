# Jetson service deployment

The unit file is an example for an installation at `/opt/ah-project` owned by
the unprivileged `ahproject` user. Adjust the carrier-board camera permissions,
paths and user before installation.

Store the RTSP URL outside the repository:

```bash
sudo install -d -m 0750 -o root -g ahproject /etc/ah-project
sudo sh -c 'printf "%s\n" "AH_RTSP_URL=rtsp://camera-host/path" > /etc/ah-project/ah-project.env'
sudo chown root:ahproject /etc/ah-project/ah-project.env
sudo chmod 0640 /etc/ah-project/ah-project.env
```

Install and start the unit after model export and runtime verification:

```bash
sudo install -m 0644 deploy/ah-project.service /etc/systemd/system/ah-project.service
sudo install -m 0644 deploy/ah-project.logrotate /etc/logrotate.d/ah-project
sudo systemctl daemon-reload
sudo systemctl enable --now ah-project.service
systemctl status ah-project.service
curl http://127.0.0.1:8000/health
```

Keep port 8000 bound to localhost. Use an authenticated TLS reverse proxy if
the dashboard must be reachable from another machine.
