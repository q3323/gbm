# Stock Backend Server Deployment

Target server path:

```text
/opt/gupaio/backend
```

Data path:

```text
/opt/gupaio/data
```

DNS record:

```text
A  api  -> 101.37.237.234
```

This backend is independent from the check-in application.

## Install

```bash
mkdir -p /opt/gupaio/backend /opt/gupaio/data
cd /opt/gupaio/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `.env`:

```text
STOCK_DATA_DIR=/opt/gupaio/data
APP_HOST=127.0.0.1
APP_PORT=8000
```

## Service

```bash
cp deploy/systemd/gupiao-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gupiao-api
systemctl status gupiao-api --no-pager
```

## HTTPS

After the DNS record resolves:

```bash
cp deploy/nginx/api.dakaapp.top.conf /etc/nginx/conf.d/
nginx -t
systemctl reload nginx
certbot --nginx -d api.dakaapp.top
```

The FastAPI backend must listen only on `127.0.0.1:8000`. Public access is
served through Nginx on port 443.

## Verify

```bash
curl http://127.0.0.1:8000/api/health
curl https://api.dakaapp.top/api/health
curl https://api.dakaapp.top/api/model/groups
```
