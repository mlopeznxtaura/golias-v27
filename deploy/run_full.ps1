# Full Golias v27 run: deploy -> secrets -> checkpoint -> train -> verify -> publish
param(
    [string]$GpuHost = "ubuntu@150.239.211.245",
    [string]$SshKey = "$env:USERPROFILE\.ssh\golias_ibm",
    [string]$RepoRoot = "",
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path }

Write-Host "=== GOLIAS FULL RUN ===" -ForegroundColor Cyan

Write-Host "`n[1/6] Deploying code to GPU..." -ForegroundColor Yellow
ssh -i $SshKey -o StrictHostKeyChecking=no $GpuHost "mkdir -p /tmp/golias-sync"
scp -i $SshKey -r "$RepoRoot\core" "$RepoRoot\training" "$RepoRoot\live" "$RepoRoot\deploy" "$RepoRoot\data" "${GpuHost}:/tmp/golias-sync/"
$deploySh = @'
set -e
sudo rsync -a /tmp/golias-sync/core/ /opt/golias-v27/core/
sudo rsync -a /tmp/golias-sync/training/ /opt/golias-v27/training/
sudo rsync -a /tmp/golias-sync/live/ /opt/golias-v27/live/
sudo rsync -a /tmp/golias-sync/data/ /opt/golias-v27/data/
sudo install -m 755 /tmp/golias-sync/deploy/load_secrets_from_sm.sh /opt/golias/load_secrets_from_sm.sh
sudo install -m 755 /tmp/golias-sync/deploy/consolidate_ckpt.sh /opt/golias-v27/deploy/consolidate_ckpt.sh
sudo install -m 755 /tmp/golias-sync/deploy/start_hybrid_train.sh /opt/golias-v27/deploy/start_hybrid_train.sh
sudo cp /tmp/golias-sync/deploy/goliasv27-dash.service /etc/systemd/system/goliasv27-dash.service
sudo cp /tmp/golias-sync/deploy/golias-secrets.service /etc/systemd/system/golias-secrets.service
sudo mkdir -p /opt/golias-v27/deploy /opt/golias-v27/checkpoints /opt/golias-v27/data/inbox /run/golias
echo deploy OK
'@
ssh -i $SshKey $GpuHost $deploySh

Write-Host "`n[2/6] Secrets (GPU secrets.env)..." -ForegroundColor Yellow

Write-Host "`n[3/6] Checkpoint + dashboard..." -ForegroundColor Yellow
ssh -i $SshKey $GpuHost "bash /opt/golias-v27/deploy/consolidate_ckpt.sh"
ssh -i $SshKey $GpuHost "sudo systemctl daemon-reload; sudo systemctl enable golias-secrets goliasv27-dash; sudo systemctl restart golias-secrets goliasv27-dash"
Start-Sleep -Seconds 4

Write-Host "`n[4/6] GitHub public + GPU gh..." -ForegroundColor Yellow
gh repo edit mlopeznxtaura/goliasv28 --visibility public --accept-visibility-change-consequences 2>$null
$ghGpu = @'
set -a
source /opt/golias/config.env 2>/dev/null || true
source /run/golias/secrets.env 2>/dev/null || true
set +a
if command -v gh >/dev/null && [ -n "${GH_TOKEN:-}" ]; then
  echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null || true
  gh config set git_protocol ssh
fi
echo gh_gpu_ok
'@
ssh -i $SshKey $GpuHost $ghGpu

if (-not $SkipTrain) {
    Write-Host "`n[5/6] Starting hybrid train..." -ForegroundColor Yellow
    ssh -i $SshKey $GpuHost "bash /opt/golias-v27/deploy/start_hybrid_train.sh"
    Start-Sleep -Seconds 8
    ssh -i $SshKey $GpuHost "tail -30 /opt/golias-v27/train_hybrid.log"
} else {
    Write-Host "`n[5/6] Train skipped" -ForegroundColor DarkYellow
}

Write-Host "`n[6/6] Live inference test..." -ForegroundColor Yellow
$infoJson = ssh -i $SshKey $GpuHost 'KEY=$(grep GOLIAS_INTERNAL_KEY /run/golias/secrets.env 2>/dev/null | cut -d= -f2-); curl -s http://127.0.0.1:8080/info -H "X-Golias-Key: $KEY"'
Write-Host $infoJson
$body = '{"geometry":0.52,"binary":0.73,"language":"move the red block to the left","m1":4.2,"m2":0.55,"m3":0.99,"V":0.58,"if7":0.5}'
$fwdJson = ssh -i $SshKey $GpuHost "KEY=`$(grep GOLIAS_INTERNAL_KEY /run/golias/secrets.env 2>/dev/null | cut -d= -f2-); curl -s -X POST http://127.0.0.1:8080/forward -H 'Content-Type: application/json' -H \"X-Golias-Key: `$KEY\" -d '$body'"
$fwd = $fwdJson | ConvertFrom-Json
Write-Host "ckpt=$($fwd.ckpt) next_frame=$($fwd.next_frame_scalar) aligned=$($fwd.outputs_aligned) mismatch=$($fwd.mismatch)"

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Dashboard: http://150.239.211.245:8080"
Write-Host "Train log: ssh -i $SshKey $GpuHost tail -f /opt/golias-v27/train_hybrid.log"
Write-Host "GitHub: https://github.com/mlopeznxtaura/goliasv28"
