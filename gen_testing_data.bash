#!/bin/bash

# Loop through task numbers from 1 to 5
# for task_number in {0..9}; do
#     # Loop through keypoint method numbers from 1 to 4
#     for keypoint_method_number in {0..3}; do
#         # Invoke the Python script with the current loop variables
#         python3 kinova_gen3_gen_testing_data.py side "$task_number" "$keypoint_method_number"
#     done
# done

# for task_number in {0..9}; do
#     # Loop through keypoint method numbers from 1 to 4
#     for keypoint_method_number in {0..3}; do
#         # Invoke the Python script with the current loop variables
#         python3 kinova_gen3_gen_testing_data.py forward "$task_number" "$keypoint_method_number"
#     done
# done

for task_number in {0..9}; do
    # Loop through keypoint method numbers from 1 to 4
    for keypoint_method_number in {0..3}; do
        # Invoke the Python script with the current loop variables
        python3 kinova_gen3_gen_testing_data.py lift "$task_number" "$keypoint_method_number"
    done
done