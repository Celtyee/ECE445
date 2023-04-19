#!/bin/bash

hidden_set=()
for i in $(seq 28 1 42); do
    # shellcheck disable=SC2206
    hidden_set+=($i)
done

rnn_set=()
for i in $(seq 2 4);do
  # shellcheck disable=SC2206
  rnn_set+=($i)
done

context_set=()
for i in $(seq 1 3);do
  context_set+=($((30*i)))
done

for hidden in "${hidden_set[@]}";do
  for rnn in "${rnn_set[@]}";do
    for context in "${context_set[@]}";do
      echo "Running with parameters: hidden=$hidden, rnn=$rnn, context=$context"
      python train.py --hidden_size "$hidden" --rnn_layers "$rnn" --context_day "$context"
      done
  done
done