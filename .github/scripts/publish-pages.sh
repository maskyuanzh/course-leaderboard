#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
if [[ $# -gt 1 ]]; then
  printf '%s\n' '用法：bash .github/scripts/publish-pages.sh [提交说明]' >&2
  exit 2
fi
if [[ $(git branch --show-current) != main ]]; then
  printf '%s\n' '请从 GitHub Pages 的 main 分支执行发布。' >&2
  exit 2
fi

started=$SECONDS
node .github/scripts/check-site.mjs
git diff --check HEAD -- index.html
printf '本地校验耗时 %s 秒。\n' "$((SECONDS - started))"

if ! git diff --quiet HEAD -- index.html; then
  if [[ $# -eq 0 ]]; then
    printf '%s\n' '请传入提交说明，以发布修改后的 index.html。' >&2
    exit 2
  fi
  git commit --only -m "$1" -- index.html
fi

revision=$(git rev-parse HEAD)
GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin HEAD:refs/heads/main
uploaded=$SECONDS
printf '已推送 %s，等待 Pages 发布。\n' "${revision:0:7}"

site_url='https://maskyuanzh.github.io/course-leaderboard/'
snapshot=$(mktemp)
trap 'rm -f -- "$snapshot"' EXIT
deadline=$((SECONDS + 600))
while (( SECONDS < deadline )); do
  if build=$(gh api 'repos/{owner}/{repo}/pages/builds/latest' --jq '[.commit, .status] | join(" ")'); then
    if [[ $build == "$revision errored" ]]; then
      gh api 'repos/{owner}/{repo}/pages/builds/latest' --jq '.error.message'
      exit 1
    fi
    if [[ $build == "$revision built" ]] &&
      curl -fsSL --connect-timeout 5 --max-time 12 "${site_url}?revision=${revision}" -o "$snapshot" &&
      git show "$revision:index.html" | cmp -s - "$snapshot"; then
      printf '已验证公网版本 %s。推送后 %s 秒，总计 %s 秒。\n%s\n' "${revision:0:7}" "$((SECONDS - uploaded))" "$((SECONDS - started))" "$site_url"
      exit 0
    fi
  fi
  sleep 4
done

printf '等待 Pages 发布超时。目标版本 %s。\n' "$revision" >&2
exit 1
