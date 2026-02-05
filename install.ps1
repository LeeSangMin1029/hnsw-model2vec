# v-hnsw Installer for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/LeeSangMin1029/v-hnsw/main/install.ps1 | iex

param(
    [switch]$Cuda,
    [string]$Version = "latest"
)

$ErrorActionPreference = "Stop"

$repo = "LeeSangMin1029/v-hnsw"
$installDir = "$env:LOCALAPPDATA\v-hnsw"

Write-Host "v-hnsw Installer" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan

# Detect architecture
$arch = if ([Environment]::Is64BitOperatingSystem) { "x86_64" } else { "i686" }
$target = "$arch-pc-windows-msvc"
$suffix = if ($Cuda) { "-cuda" } else { "" }

# Get latest release or specific version
if ($Version -eq "latest") {
    Write-Host "Fetching latest release..."
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest"
    $Version = $release.tag_name
} else {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/tags/$Version"
}

Write-Host "Installing v-hnsw $Version ($target$suffix)..." -ForegroundColor Green

# Find matching asset
$assetName = "v-hnsw-$target$suffix.zip"
$asset = $release.assets | Where-Object { $_.name -eq $assetName }

if (-not $asset) {
    Write-Host "Error: Could not find release asset: $assetName" -ForegroundColor Red
    Write-Host "Available assets:" -ForegroundColor Yellow
    $release.assets | ForEach-Object { Write-Host "  - $($_.name)" }
    exit 1
}

# Create install directory
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Download
$zipPath = "$env:TEMP\v-hnsw.zip"
Write-Host "Downloading from $($asset.browser_download_url)..."
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath

# Extract
Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
Remove-Item $zipPath

# Add to PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$installDir*") {
    Write-Host "Adding to PATH..."
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$installDir", "User")
    $env:Path = "$env:Path;$installDir"
}

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Run 'v-hnsw --help' to get started." -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: Restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
