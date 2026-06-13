# Deploy M1/M2/M3 sidecars to IBM (GPU + CE fallback apps + golias-live rebuild)
param(
    [string]$GpuHost = "ubuntu@150.239.211.245",
    [string]$SshKey = "$env:USERPROFILE\.ssh\golias_ibm",
    [string]$CeProject = "golias-hybrid",
    [string]$CeRegion = "us-east",
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent)
)

$ErrorActionPreference = "Stop"

Write-Host "=== Sync golias-v27 to GPU ===" -ForegroundColor Cyan
ssh -i $SshKey -o StrictHostKeyChecking=no $GpuHost "mkdir -p /tmp/golias-v27-sync"
scp -i $SshKey -r "$RepoRoot\live" "$RepoRoot\core" "$RepoRoot\sidecars" "$RepoRoot\training" "$RepoRoot\data" ubuntu@150.239.211.245:/tmp/golias-v27-sync/

ssh -i $SshKey -o StrictHostKeyChecking=no $GpuHost @'
sudo rsync -a /tmp/golias-v27-sync/live/ /opt/golias-v27/live/
sudo rsync -a /tmp/golias-v27-sync/core/ /opt/golias-v27/core/
sudo rsync -a /tmp/golias-v27-sync/sidecars/ /opt/golias-v27/sidecars/
sudo rsync -a /tmp/golias-v27-sync/training/ /opt/golias-v27/training/
sudo rsync -a /tmp/golias-v27-sync/data/ /opt/golias-v27/data/
sudo systemctl restart goliasv27-dash
sleep 2
curl -s http://127.0.0.1:8080/health
'@

Write-Host "`n=== CE: rebuild golias-live from GitHub ===" -ForegroundColor Cyan
ibmcloud target -g Default | Out-Null
ibmcloud target -r $CeRegion | Out-Null
ibmcloud ce project select -n $CeProject | Out-Null

ibmcloud ce app update -n golias-live `
    --build-source https://github.com/mlopeznxtaura/golias-v27 `
    --build-dockerfile Dockerfile.proxy `
    --build-strategy dockerfile `
    --rebuild `
    --env-from-secret golias-gpu-url `
    --env-from-secret golias-internal-key `
    --command python `
    --argument /app/live/proxy_dashboard.py

foreach ($role in @("m1", "m2", "m3")) {
    $app = "golias-if-$role"
    Write-Host "`n=== CE app $app (min-scale 0) ===" -ForegroundColor Cyan
    ibmcloud ce app get -n $app 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Creating $app (requires CE create permission)..." -ForegroundColor Yellow
        ibmcloud ce app create -n $app `
            --build-source https://github.com/mlopeznxtaura/golias-v27 `
            --build-dockerfile Dockerfile.if `
            --build-strategy dockerfile `
            --env IF_ROLE=$role `
            --min-scale 0 `
            --max-scale 1 `
            --cpu 0.25 `
            --memory 0.5G `
            --port 8080
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  SKIP $app — use GPU local-fallback until IAM allows CE app create" -ForegroundColor Red
            continue
        }
    } else {
        ibmcloud ce app update -n $app `
            --build-source https://github.com/mlopeznxtaura/golias-v27 `
            --build-dockerfile Dockerfile.if `
            --rebuild `
            --env IF_ROLE=$role `
            --min-scale 0 `
            --max-scale 1
    }
    $url = (ibmcloud ce app get -n $app --output json | ConvertFrom-Json).url
    Write-Host "  $app -> $url"
}

Write-Host "`nSet on GPU /opt/golias/env.sh (or systemd drop-in):" -ForegroundColor Yellow
Write-Host "  IF_BACKEND=watsonx"
Write-Host "  WATSONX_API_KEY=... WATSONX_PROJECT_ID=... WATSONX_URL=https://us-south.ml.cloud.ibm.com"
Write-Host "  IF_FALLBACK_M1=https://golias-if-m1..../invoke (use CE app URLs)"
