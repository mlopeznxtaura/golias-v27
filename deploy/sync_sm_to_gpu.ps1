# Pull Watsonx secrets from IBM SM → GPU /opt/golias/env.sh (never touches website/repo).
param(
    [string]$GpuHost = "ubuntu@150.239.211.245",
    [string]$SshKey = "$env:USERPROFILE\.ssh\golias_ibm",
    [string]$SmUrl = "https://77a74a8e-30d4-440a-b4da-7eb56ff43425.us-south.secrets-manager.appdomain.cloud",
    [string]$ApiKeySecretName = "watsonx-api-key-golias",
    [string]$ProjectSecretName = "watsonx-project-id-golias"
)

$ErrorActionPreference = "Stop"
$env:SECRETS_MANAGER_URL = $SmUrl

function Get-SmPayload($Name) {
    $json = ibmcloud secrets-manager secret-by-name --name $Name --secret-type arbitrary --secret-group-name default --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        $json = ibmcloud secrets-manager secret --name $Name --secret-type arbitrary --output json
    }
    $obj = ($json | Out-String) | ConvertFrom-Json
    if ($obj.payload) { return $obj.payload }
    if ($obj.data -and $obj.data.payload) { return $obj.data.payload }
    throw "Could not read payload for secret $Name"
}

Write-Host "Reading secrets from IBM Secrets Manager..." -ForegroundColor Cyan
$apiKey = Get-SmPayload $ApiKeySecretName
$projectId = Get-SmPayload $ProjectSecretName

$envBlock = @"
IF_BACKEND=watsonx
WATSONX_API_KEY=$apiKey
WATSONX_PROJECT_ID=$projectId
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL=ibm/granite-3-8b-instruct
"@

$tmp = New-TemporaryFile
$envBlock | Set-Content -Path $tmp -NoNewline
scp -i $SshKey $tmp.FullName "${GpuHost}:/tmp/watsonx-env.fragment"
Remove-Item $tmp -Force

ssh -i $SshKey -o StrictHostKeyChecking=no $GpuHost @'
set -e
sudo touch /opt/golias/env.sh
sudo sed -i '/^IF_BACKEND=/d;/^WATSONX_/d' /opt/golias/env.sh
sudo tee -a /opt/golias/env.sh < /tmp/watsonx-env.fragment > /dev/null
sudo chmod 640 /opt/golias/env.sh
rm -f /tmp/watsonx-env.fragment
sudo systemctl restart goliasv27-dash
sleep 2
curl -s http://127.0.0.1:8080/health
'@

Write-Host "Done. Secrets on GPU only in /opt/golias/env.sh (not in git or CE UI)." -ForegroundColor Green
