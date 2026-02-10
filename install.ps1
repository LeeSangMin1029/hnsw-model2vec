# v-hnsw installer for Windows
# Downloads pre-built binary from GitHub Release.
# Requires: git clone access to this repo (credentials reused automatically).

$ErrorActionPreference = "Stop"

$repo = "LeeSangMin1029/hnsw-model2vec"
$assetPattern = "v-hnsw-x86_64-pc-windows-msvc"
$installDir = "$env:LOCALAPPDATA\v-hnsw"
$binPath = "$installDir\v-hnsw.exe"

Write-Host "=== v-hnsw installer ===" -ForegroundColor Cyan

# --- 1. Get GitHub token from Git Credential Manager ---
Write-Host "Authenticating via Git credentials..." -ForegroundColor Cyan
$credInput = "protocol=https`nhost=github.com`n`n"
$credOutput = $credInput | git credential fill 2>$null

$token = ""
foreach ($line in $credOutput -split "`n") {
    if ($line -match "^password=(.+)$") {
        $token = $Matches[1].Trim()
    }
}

if (-not $token) {
    Write-Host "ERROR: Could not retrieve GitHub credentials." -ForegroundColor Red
    Write-Host "Make sure you can 'git clone' this repo first." -ForegroundColor Red
    exit 1
}

# --- 2. Get latest release ---
Write-Host "Fetching latest release..." -ForegroundColor Cyan
$headers = @{
    "Authorization" = "token $token"
    "User-Agent"    = "v-hnsw-installer"
    "Accept"        = "application/vnd.github+json"
}

$releaseUrl = "https://api.github.com/repos/$repo/releases/latest"
try {
    $release = Invoke-RestMethod -Uri $releaseUrl -Headers $headers
} catch {
    Write-Host "ERROR: No releases found. Ask the maintainer to create a release." -ForegroundColor Red
    exit 1
}

$tag = $release.tag_name
Write-Host "Found release: $tag" -ForegroundColor Cyan

$asset = $release.assets | Where-Object { $_.name -like "*$assetPattern*" }
if (-not $asset) {
    Write-Host "ERROR: Windows binary not found in release $tag." -ForegroundColor Red
    exit 1
}

# --- 3. Download and extract ---
Write-Host "Downloading $($asset.name)..." -ForegroundColor Cyan

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

$zipPath = "$installDir\v-hnsw-download.zip"
$dlHeaders = @{
    "Authorization" = "token $token"
    "User-Agent"    = "v-hnsw-installer"
    "Accept"        = "application/octet-stream"
}
Invoke-WebRequest -Uri $asset.url -Headers $dlHeaders -OutFile $zipPath

Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
Remove-Item $zipPath
Write-Host "Installed to: $binPath" -ForegroundColor Green

# --- 4. Add to user PATH if not already present ---
$userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$installDir*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$userPath;$installDir", "User")
    $env:PATH = "$env:PATH;$installDir"
    Write-Host "Added $installDir to user PATH." -ForegroundColor Green
}

# --- 5. Verify ---
Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host "Binary: $binPath"
& $binPath --version
Write-Host ""
Write-Host "Restart your terminal, then run 'v-hnsw --help' to get started."
Write-Host "First run will auto-download the embedding model (~500MB) and Korean dictionary."
