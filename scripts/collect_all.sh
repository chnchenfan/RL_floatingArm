run(){ MB_R=$1 MB_T=$2 MB_Z=$3 RUN=38 timeout 110 bash scripts/collect_mpc_v3.sh >/dev/null 2>&1; echo "  done R=$1 T=$2 Z=$3"; sleep 3; }
echo "[all] collecting 3 configs x2..."
run 0.02 1.0 0.0
run 0.02 1.0 0.02
run 0.02 1.0 0.02
run 0.01 0.5 0.01
run 0.01 0.5 0.01
echo "[all] DONE"; ls -1t /home/windylab/code/windylab_ws/src/arm-platform/demo/data/moving_base_mpc_loop_*.csv | head -6
