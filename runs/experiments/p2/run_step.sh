#!/bin/zsh
# chạy 1 bước thang, tự tắt sau 40 phút (yêu cầu của người dùng)
cd "$(dirname "$0")/../../.."
step=$1; shift
caffeinate -i .venv/bin/python runs/experiments/p2/ladder.py "$step" "$@" > "runs/experiments/p2/$step.log" 2>&1 &
pid=$!
( sleep 2400; kill $pid 2>/dev/null && echo "KILLED $step after 40 min" >> "runs/experiments/p2/$step.log" ) &
wd=$!
wait $pid; rc=$?
kill $wd 2>/dev/null
grep -v "^[WI]0000" "runs/experiments/p2/$step.log" | tail -5
exit $rc
