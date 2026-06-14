#!/bin/bash
# Generate GitHub SSH deploy key on GPU (same pattern as Mac SSH key in GitHub Settings).
set -euo pipefail

KEY="$HOME/.ssh/id_ed25519_golias_gpu"
SSH_CONFIG="$HOME/.ssh/config"
ENV_FILE="${GOLIAS_ENV:-/opt/golias/env.sh}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$KEY" ]]; then
  ssh-keygen -t ed25519 -C "golias-gpu-ibm" -f "$KEY" -N ""
  echo "Created new Ed25519 key: $KEY"
else
  echo "Key already exists: $KEY"
fi

# GitHub host block — dedicated key for Golias GPU pushes
if ! grep -q 'Host github.com-golias' "$SSH_CONFIG" 2>/dev/null; then
  cat >> "$SSH_CONFIG" <<EOF

Host github.com-golias
  HostName github.com
  User git
  IdentityFile $KEY
  IdentitiesOnly yes
EOF
  chmod 600 "$SSH_CONFIG"
fi

# Default github.com → golias key (so git push and gh use same key)
if ! grep -q '^Host github.com$' "$SSH_CONFIG" 2>/dev/null; then
  cat >> "$SSH_CONFIG" <<EOF

Host github.com
  HostName github.com
  User git
  IdentityFile $KEY
  IdentitiesOnly yes
EOF
fi

chmod 600 "$KEY"
chmod 644 "${KEY}.pub"

# gh + git prefer SSH (matches Mac workflow)
if command -v gh >/dev/null; then
  gh config set git_protocol ssh
fi
git config --global url."git@github.com:".insteadOf "https://github.com/" 2>/dev/null || true

# Non-secret defaults only
if [[ -f "$ENV_FILE" ]]; then
  grep -q '^GOLIAS_GH_VISIBILITY=' "$ENV_FILE" 2>/dev/null || \
    echo 'GOLIAS_GH_VISIBILITY=public' | sudo tee -a "$ENV_FILE" >/dev/null
fi

echo ""
echo "========== ADD THIS KEY TO GITHUB =========="
echo "https://github.com/settings/ssh/new"
echo "Title: Golias GPU IBM"
echo "Key type: Authentication Key"
echo ""
cat "${KEY}.pub"
echo ""
echo "Fingerprint (SHA256):"
ssh-keygen -lf "${KEY}.pub"
echo "============================================"
echo ""
echo "GitHub org/user: set GOLIAS_GH_ORG in IBM SM (github-org-golias), then sync_all_sm_to_gpu.ps1"
echo "Test: ssh -T git@github.com"
