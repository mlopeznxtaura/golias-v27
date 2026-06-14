# Pull ALL Golias secrets from IBM Secrets Manager → GPU (no hardcoded orgs/tokens/URLs in repo).
# Prereqs: ibmcloud CLI + secrets-manager plugin, logged in.
#
#   $env:IBM_SECRETS_MANAGER_URL = 'https://<instance-id>.<region>.secrets-manager.appdomain.cloud'
#   $env:GPU_HOST = 'ubuntu@<your-gpu-ip>'
#   $env:SSH_KEY = "$env:USERPROFILE\.ssh\golias_ibm"
#   .\deploy\sync_all_sm_to_gpu.ps1
param(
    [string]$GpuHost = $(if ($env:GPU_HOST) { $env:GPU_HOST } else { throw "Set GPU_HOST" }),
    [string]$SshKey = $(if ($env:SSH_KEY) { $env:SSH_KEY } else { "$env:USERPROFILE\.ssh\golias_ibm" }),
    [string]$SmUrl = $(if ($env:IBM_SECRETS_MANAGER_URL) { $env:IBM_SECRETS_MANAGER_URL } else { throw "Set IBM_SECRETS_MANAGER_URL" }),
    [string]$ManifestPath = "$PSScriptRoot\sm_manifest.json"
)

$ErrorActionPreference = "Stop"
$env:SECRETS_MANAGER_URL = $SmUrl

if (-not (Test-Path $ManifestPath)) { throw "Missing $ManifestPath" }
$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$group = $manifest.secret_group

function Get-SmPayload($Name) {
    $json = ibmcloud secrets-manager secret-by-name --name $Name --secret-type arbitrary --secret-group-name $group --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        $json = ibmcloud secrets-manager secret --name $Name --secret-type arbitrary --output json
    }
    $obj = ($json | Out-String) | ConvertFrom-Json
    if ($obj.payload) { return $obj.payload }
    if ($obj.data -and $obj.data.payload) { return $obj.data.payload }
    throw "Could not read SM secret: $Name"
}

Write-Host "IBM SM → GPU sync ($GpuHost)" -ForegroundColor Cyan

$secretLines = @()
foreach ($row in $manifest.secrets) {
    $val = Get-SmPayload $row.sm_name
    $secretLines += "$($row.env)=$val"
    Write-Host "  OK $($row.sm_name) → $($row.env)" -ForegroundColor Green
}

$configLines = @(
    "IBM_SECRETS_MANAGER_URL=$SmUrl",
    "GOLIAS_ROOT=/opt/golias-v27",
    "GOLIAS_CONFIG_FILE=/opt/golias/config.env",
    "GOLIAS_SECRETS_FILE=/run/golias/secrets.env"
)
foreach ($prop in $manifest.config.PSObject.Properties) {
    $configLines += "$($prop.Name)=$($prop.Value)"
}

$secTmp = New-TemporaryFile
$cfgTmp = New-TemporaryFile
($secretLines -join "`n") | Set-Content -Path $secTmp -NoNewline
($configLines -join "`n") | Set-Content -Path $cfgTmp -NoNewline

scp -i $SshKey $secTmp.FullName "${GpuHost}:/tmp/golias-secrets.env"
scp -i $SshKey $cfgTmp.FullName "${GpuHost}:/tmp/golias-config.env"
Remove-Item $secTmp, $cfgTmp -Force

$readerKey = Get-SmPayload "ibm-cloud-api-key-golias"

scp -i $SshKey "$PSScriptRoot\load_secrets_from_sm.sh" "${GpuHost}:/tmp/load_secrets_from_sm.sh"

$sshScript = @"
set -e
sudo mkdir -p /run/golias /opt/golias
sudo mv /tmp/golias-secrets.env /run/golias/secrets.env
sudo chmod 600 /run/golias/secrets.env
sudo mv /tmp/golias-config.env /opt/golias/config.env
sudo chmod 644 /opt/golias/config.env
echo '$readerKey' | sudo tee /opt/golias/.ibm_cloud_api_key >/dev/null
sudo chmod 600 /opt/golias/.ibm_cloud_api_key
sudo install -m 755 /tmp/load_secrets_from_sm.sh /opt/golias/load_secrets_from_sm.sh
if [ -f /opt/golias/env.sh ]; then
  sudo sed -i '/^GH_TOKEN=/d;/^GITHUB_TOKEN=/d;/^WATSONX_/d;/^HF_TOKEN=/d;/^GOLIAS_INTERNAL_KEY=/d;/^IBM_CLOUD_API_KEY=/d;/^GOLIAS_GH_ORG=/d' /opt/golias/env.sh
fi
sudo systemctl restart goliasv27-dash 2>/dev/null || true
echo OK
"@
ssh -i $SshKey -o StrictHostKeyChecking=no $GpuHost $sshScript

Write-Host "Done. Secrets in /run/golias/secrets.env only (not git, not env.sh)." -ForegroundColor Green
