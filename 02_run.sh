#!/bin/bash

# List of dataset/run_id pairs
declare -a runs=(
#  "mnist run-0011-CNN_mnist_32_0.9776"
#  "mnist run-0014-CNN_mnist_32_0.9744"
  "mnist run-0030-ViT_mnist_32_0.9812"
  "cifar10 run-0016-CNN_cifar10_128_0.8093"
  "cifar10 run-0023-CNN_cifar10_128_0.8509"
  "cifar10 run-0036-ViT_cifar10_32_0.6299"
  "cifar10 run-0041-ViT_cifar10_256_0.8107"
)

# Loop over each run
for entry in "${runs[@]}"; do
  set -- $entry
  dataset=$1
  run_id=$2

  echo "Running evaluation for $run_id ($dataset)"
  python 02_automatic_evaluation.py --dataset "$dataset" --run_id "$run_id"
done