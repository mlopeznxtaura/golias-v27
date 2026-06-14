# Store Watsonx credentials in IBM Secrets Manager (never commit secrets).
# Usage:
#   $env:WATSONX_API_KEY = '<api key from iam api-key-create>'
#   .\deploy\store_watsonx_sm.ps1
# Optional project id (skip auto-detect):
#   $env:WATSONX_PROJECT_ID = '<project id>'
param(
    [string]$SmUrl = $(if ($env:IBM_SECRETS_MANAGER_URL) { $env:IBM_SECRETS_MANAGER_URL } else { throw "Set IBM_SECRETS_MANAGER_URL" }),
    [string]$ApiKeySecretName = "watsonx-api-key-golias",
    [string]$ProjectSecretName = "watsonx-project-id-golias"
)

$ErrorActionPreference = "Stop"
$env:SECRETS_MANAGER_URL = $SmUrl

if (-not $env:WATSONX_API_KEY) {
    throw "Set WATSONX_API_KEY in your shell first (do not paste into git or UI code)."
}

function Ensure-ArbitrarySecret($Name, $Payload, $Description) {
    $search = ibmcloud secrets-manager secrets --search $Name --all-pages --output json 2>&1
    $parsed = $search | ConvertFrom-Json -ErrorAction SilentlyContinue
    $existing = $parsed.secrets | Where-Object { $_.name -eq $Name }
    if ($existing) {
        Write-Host "Secret exists, adding new version: $Name" -ForegroundColor Yellow
        $id = $existing[0].id
        ibmcloud secrets-manager secret-version-create --secret-id $id --arbitrary-payload $Payload | Out-Null
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

Ensure-ArbitrarySecret $ApiKeySecretName $env:WATSONX_API_KEY "Golias M1/M2/M3 Watsonx API key"

if ($env:WATSONX_PROJECT_ID) {
    Ensure-ArbitrarySecret $ProjectSecretName $env:WATSONX_PROJECT_ID "Watsonx project ID for Golias sidecars"
    exit 0
}

Write-Host "Fetching Watsonx project ID..." -ForegroundColor Cyan
$tokenBody = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=$($env:WATSONX_API_KEY)"
$token = Invoke-RestMethod -Method Post -Uri "https://iam.cloud.ibm.com/identity/token" `
    -ContentType "application/x-www-form-urlencoded" -Body $tokenBody
$headers = @{ Authorization = "Bearer $($token.access_token)" }
$endpoints = @(
    "https://api.dataplatform.cloud.ibm.com/v2/projects",
    "https://us-south.ml.cloud.ibm.com/ml/v1/projects?version=2023-05-29"
)
$projects = $null
foreach ($uri in $endpoints) {
    try {
        $projects = Invoke-RestMethod -Method Get -Uri $uri -Headers $headers
        if ($projects) { break }
    } catch { }
}
$projectId = $null
if ($projects.resources) { $projectId = $projects.resources[0].metadata.guid }
elseif ($projects.results) { $projectId = $projects.results[0].metadata.guid }
elseif ($projects -is [array] -and $projects.Count -gt 0) { $projectId = $projects[0].guid }
if (-not $projectId) {
    throw "No Watsonx projects found. Set WATSONX_PROJECT_ID manually and re-run."
}
Write-Host "Using project id: $projectId"
Ensure-ArbitrarySecret $ProjectSecretName $projectId "Watsonx project ID for Golias sidecars"
