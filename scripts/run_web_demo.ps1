param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$env:BRAIN_TUMOR_WEB_USER = "admin"
$env:BRAIN_TUMOR_WEB_PASSWORD = "brain2026"

python -m uvicorn brain_tumor_fl.web_app:app --host $HostAddress --port $Port
