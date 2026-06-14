# Rebuild golias-live CE app from GitHub (new UI in ui_page.py)
param(
    [string]$CeProject = "golias-hybrid",
    [string]$CeRegion = "us-east",
    [string]$Repo = "https://github.com/mlopeznxtaura/golias-v27"
)

$ErrorActionPreference = "Stop"
Write-Host "=== Rebuild golias-live on Code Engine ===" -ForegroundColor Cyan

ibmcloud target -g Default | Out-Null
ibmcloud target -r $CeRegion | Out-Null
ibmcloud ce project select -n $CeProject | Out-Null

ibmcloud ce app update -n golias-live `
    --build-source $Repo `
    --build-dockerfile Dockerfile.proxy `
    --build-strategy dockerfile `
    --rebuild `
    --env-from-secret golias-gpu-url `
    --env-from-secret golias-internal-key `
    --command python `
    --argument /app/live/proxy_dashboard.py

$url = (ibmcloud ce app get -n golias-live --output json | ConvertFrom-Json).url
Write-Host "`nLive UI: $url" -ForegroundColor Green
