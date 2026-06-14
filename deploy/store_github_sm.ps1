# Store GitHub credentials in IBM Secrets Manager (never commit tokens or org names to git).
# Usage:
#   $env:IBM_SECRETS_MANAGER_URL = 'https://<instance>.<region>.secrets-manager.appdomain.cloud'
#   $env:GH_TOKEN = '<fine-grained PAT: repo create + contents write>'
#   $env:GOLIAS_GH_ORG = '<github username or org>'   # not hardcoded in repo
#   .\deploy\store_github_sm.ps1
param(
    [string]$SmUrl = $(if ($env:IBM_SECRETS_MANAGER_URL) { $env:IBM_SECRETS_MANAGER_URL } else { throw "Set IBM_SECRETS_MANAGER_URL" }),
    [string]$TokenSecretName = "github-token-golias",
    [string]$OrgSecretName = "github-org-golias"
)

$ErrorActionPreference = "Stop"
$env:SECRETS_MANAGER_URL = $SmUrl

if (-not $env:GH_TOKEN) { throw "Set GH_TOKEN in your shell (do not commit)." }
if (-not $env:GOLIAS_GH_ORG) { throw "Set GOLIAS_GH_ORG (your GitHub username or org)." }

function Ensure-ArbitrarySecret($Name, $Payload, $Description) {
    $search = ibmcloud secrets-manager secrets --search $Name --all-pages --output json 2>&1
    $parsed = $search | ConvertFrom-Json -ErrorAction SilentlyContinue
    $existing = $parsed.secrets | Where-Object { $_.name -eq $Name }
    if ($existing) {
        Write-Host "Secret exists, new version: $Name" -ForegroundColor Yellow
        ibmcloud secrets-manager secret-version-create --secret-id $existing[0].id --arbitrary-payload $Payload | Out-Null
    } else {
        ibmcloud secrets-manager secret-create `
            --secret-name $Name `
            --secret-type arbitrary `
            --secret-group-id default `
            --secret-description $Description `
            --arbitrary-payload $Payload | Out-Null
    }
    Write-Host "OK $Name" -ForegroundColor Green
}

Ensure-ArbitrarySecret $TokenSecretName $env:GH_TOKEN "Golias GitHub PAT (public repo publish)"
Ensure-ArbitrarySecret $OrgSecretName $env:GOLIAS_GH_ORG "Golias GitHub org/user for checkpoint repos"
Write-Host "Run: .\deploy\sync_all_sm_to_gpu.ps1" -ForegroundColor Cyan
