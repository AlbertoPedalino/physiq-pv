#!/usr/bin/env bash
# Memory of the server and of this user's processes, every INTERVAL seconds (default 300), appended to the file
# given (default outputs/memory.log): the free memory, the use of /dev/shm, every process of this user above
# 100 MB with its anonymous pages (its own memory), file pages (maps of data files, which the kernel can drop)
# and shared memory, from /proc/<pid>/status, and the five largest processes of the server, of any user. When
# the out-of-memory killer stops a run, the last lines say who was filling the memory.
#
#     bash scripts/log_memory.sh outputs/optuna/memory.log
set -u
interval=${INTERVAL:-300}
out=${1:-outputs/memory.log}
mkdir -p "$(dirname "$out")"
while true; do
  {
    echo "=== $(date '+%F %T')"
    free -m | sed -n 1,3p
    df -m /dev/shm | tail -1
    echo "pid name anon_MB file_MB shmem_MB (this user, above 100 MB)"
    for pid in $(pgrep -u "$(id -u)"); do
      [ -r "/proc/$pid/status" ] || continue
      awk -v pid="$pid" '/^Name:/{name=$2} /^VmRSS:/{rss=$2} /^RssAnon:/{anon=$2} /^RssFile:/{file=$2}
        /^RssShmem:/{shmem=$2}
        END{if (rss > 102400) printf "%s %s %d %d %d\n", pid, name, anon/1024, file/1024, shmem/1024}' \
        "/proc/$pid/status" 2>/dev/null
    done
    echo "largest processes of the server (RSS MB):"
    ps -eo user:16,pid,rss,comm --sort=-rss | awk 'NR == 1 {print; next} NR <= 6 {$3 = int($3 / 1024); print}'
  } >> "$out"
  sleep "$interval"
done
