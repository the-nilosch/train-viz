#!/bin/bash

# List of dataset/run_id pairs
declare -a runs=(
  "mnist run-0011-CNN_mnist_32_0.9776"
  "mnist run-0014-CNN_mnist_32_0.9744"
  "mnist run-0030-ViT_mnist_32_0.9812"
  "cifar10 run-0016-CNN_cifar10_128_0.8093"
  "cifar10 run-0023-CNN_cifar10_128_0.8509"
  "cifar10 run-0036-ViT_cifar10_32_0.6299"
  "cifar10 run-0041-ViT_cifar10_256_0.8107"

  "mnist run-0012-CNN_mnist_32_0.9768"
  "mnist run-0013-CNN_mnist_32_0.9797"

  "cifar10 run-0016-CNN_cifar10_128_0.8093"
  "cifar10 run-0017-CNN_cifar10_128_0.8072"
  "cifar10 run-0018-CNN_cifar10_128_0.8499"
  "cifar10 run-0019-CNN_cifar10_128_0.8487"
  "cifar10 run-0020-CNN_cifar10_128_0.8079"
  "cifar10 run-0021-CNN_cifar10_128_0.8054"
  "cifar10 run-0022-CNN_cifar10_128_0.8519"
  "cifar10 run-0023-CNN_cifar10_128_0.8509"
  "cifar10 run-0024-CNN_cifar10_128_0.8062"
  "cifar10 run-0025-CNN_cifar10_128_0.8062"
  "cifar10 run-0026-CNN_cifar10_128_0.8504"
  "cifar10 run-0027-CNN_cifar10_128_0.8503"

  "cifar10 run-0039-ViT_cifar10_256_0.7291"
)

# Loop over each run
for entry in "${runs[@]}"; do
  set -- $entry
  dataset=$1
  run_id=$2

  echo "Running evaluation for $run_id ($dataset)"
  python 02_automatic_evaluation.py --dataset "$dataset" --run_id "$run_id"
done