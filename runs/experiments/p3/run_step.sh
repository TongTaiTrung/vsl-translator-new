#!/bin/zsh
# chạy 1 bước thang: 3 seed song song (mỗi seed 1 process), tự tắt sau 40 phút (yêu cầu của người dùng)
cd "$(dirname "$0")/../../.."
step=$1; shift
pids=()
for seed in 42 7 1; do
  caffeinate -i .venv/bin/python runs/experiments/p3/ladder.py "$step" --seeds $seed "$@" \
    > "runs/experiments/p3/${step}_s$seed.log" 2>&1 &
  pids+=($!)
done
( sleep 2400; for p in $pids; do kill $p 2>/dev/null && echo "KILLED $step after 40 min"; done ) &
wd=$!
rc=0
for p in $pids; do wait $p || rc=1; done
kill $wd 2>/dev/null
for seed in 42 7 1; do grep -v "^[WI]0000" "runs/experiments/p3/${step}_s$seed.log" | tail -2; done
exit $rc
