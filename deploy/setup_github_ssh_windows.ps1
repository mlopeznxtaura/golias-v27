# Windows: GitHub SSH setup (Ed25519 + ssh-agent + git/gh over SSH).
# Run in PowerShell. Admin required once for ssh-agent service.
param(
    [string]$Email = "golias-windows-github",
    [string]$KeyName = "id_ed25519_github"
)

$ErrorActionPreference = "Stop"
$sshDir = Join-Path $env:USERPROFILE ".ssh"
$keyPath = Join-Path $sshDir $KeyName
$pubPath = "$keyPath.pub"
$winSsh = "C:/Windows/System32/OpenSSH/ssh.exe"

New-Item -ItemType Directory -Force -Path $sshDir | Out-Null

if (-not (Test-Path $keyPath)) {
    ssh-keygen -t ed25519 -C $Email -f $keyPath -N '""'
    Write-Host "Created $keyPath" -ForegroundColor Green
} else {
    Write-Host "Key exists: $keyPath" -ForegroundColor Yellow
}

# ssh-agent (needs elevated PowerShell once)
$agent = Get-Service ssh-agent -ErrorAction SilentlyContinue
if ($agent -and $agent.Status -ne "Running") {
    try {
        Set-Service ssh-agent -StartupType Manual
        Start-Service ssh-agent
        ssh-add $keyPath
        Write-Host "ssh-agent running, key added" -ForegroundColor Green
    } catch {
        Write-Warning @"
ssh-agent needs Admin once. In elevated PowerShell:
  Get-Service ssh-agent | Set-Service -StartupType Manual
  Start-Service ssh-agent
  ssh-add $keyPath
Key has no passphrase — SSH works without agent too.
"@
    }
}

# ~/.ssh/config — github.com uses this key; IBM GPU uses golias_ibm
$sshDirSlash = ($sshDir -replace '\\','/')
$keySlash = "$sshDirSlash/$KeyName"
$configPath = Join-Path $sshDir "config"
$githubBlock = @"
Host github.com
    HostName github.com
    User git
    IdentityFile $keySlash
    IdentitiesOnly yes

"@
$existing = if (Test-Path $configPath) { Get-Content $configPath -Raw } else { "" }
if ($existing -notmatch "Host github\.com") {
    Set-Content -Path $configPath -Value ($githubBlock + $existing) -Encoding utf8
} else {
  (Get-Content $configPath) -replace 'IdentityFile.*id_ed25519.*', "    IdentityFile $($keyPath -replace '\\','/')" | Set-Content $configPath
}

if (Test-Path (Join-Path $sshDir "golias_ibm")) {
    $gpuKey = Join-Path $sshDir "golias_ibm"
    $gpuSlash = ($gpuKey -replace '\\','/')
    $gpuBlock = @"

Host 150.239.211.245
    HostName 150.239.211.245
    User ubuntu
    IdentityFile $gpuSlash
    IdentitiesOnly yes
"@
    if ((Get-Content $configPath -Raw) -notmatch "150\.239\.211\.245") {
        Add-Content -Path $configPath -Value $gpuBlock
    }
}

git config --global core.sshCommand $winSsh
gh config set git_protocol ssh

Write-Host "`nFingerprint:" -ForegroundColor Cyan
ssh-keygen -lf $pubPath

Write-Host "`nPublic key (add at https://github.com/settings/ssh/new if gh add fails):" -ForegroundColor Cyan
Get-Content $pubPath

try {
    gh auth refresh -h github.com -s admin:public_key 2>$null
    gh ssh-key add $pubPath --title "Golias Windows SSH"
    Write-Host "Key added to GitHub via gh" -ForegroundColor Green
} catch {
    Write-Warning "Run: gh auth refresh -h github.com -s admin:public_key"
    Write-Warning "Then: gh ssh-key add `"$pubPath`" --title `"Golias Windows SSH`""
}

Write-Host "`nTest:" -ForegroundColor Cyan
& $winSsh -T git@github.com 2>&1
