#!/bin/bash
B=$(dirname $0)
Q=Qwen/Qwen3-Embedding-0.6B; G=Qwen/Qwen3-Embedding-0.6B-GGUF
for r in 1 2 3; do
  for v in "a_candle_cpu|$Q|cpu" "b_candle_metal|$Q|metal" "c_q8_cpu|$G:Q8_0|cpu" "d_q8_metal|$G:Q8_0|metal" "e_f16_cpu|$G:F16|cpu" "f_f16_metal|$G:F16|metal"; do
    IFS='|' read n m d <<<"$v"
    echo "load $(uptime | sed 's/.*averages: //')" > $B/runs/${n}_$r.load
    $B/run1.sh "$m" $d $B/runs/${n}_$r
  done
done
echo finished > $B/matrix.done
