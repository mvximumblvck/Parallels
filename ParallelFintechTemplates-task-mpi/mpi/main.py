from argparse import ArgumentParser
from mpi4py import MPI
import time

# import MPI

import pathlib
from pathlib import Path

import numpy as np
import os

import json


def parse_args():
    parser = ArgumentParser('Process files')
    parser.add_argument('--mode', required=True, choices=['sequence', 'parallel'])
    return parser.parse_args()


def read_files(from_file: int, to_file: int):

    batch_values = []
    for index in range(from_file, to_file):
        path_file = Path(pathlib.Path.cwd(), 'files', os.listdir('files')[index])
        file = open(path_file, 'r')
        for number in file.readlines():
            batch_values.append(float(number))
        file.close()
    return np.array(batch_values, dtype=np.float32)

def calculate_statistics(client_sums):

    sums = np.sum(client_sums, dtype=np.float32)
    sums_squares = np.sum(client_sums ** 2, dtype=np.float32)

    below_1M = (client_sums <= 1_000_000).sum()
    below_10M = (client_sums <= 10_000_000).sum()
    below_100M = (client_sums <= 100_000_000).sum()
    below_1B = (client_sums <= 1_000_000_000).sum()
    more_1B = (client_sums > 1_000_000_000).sum()

    count = below_1B + more_1B
    count_1M_10M = below_10M - below_1M
    count_10M_100M = below_100M - below_10M
    count_100M_1B = below_1B - below_100M
    return np.array([{
        'count': count,
        '1M': below_1M,
        '10M': count_1M_10M,
        '100M': count_10M_100M,
        '1B': count_100M_1B,
        '1B+': more_1B,
        'sum': sums,
        'sum_squares': sums_squares
    }])


def accumulate_stats(stats):

    all_count = sum([stats[i][0]['count'] for i in range(len(stats))])
    all_sums = sum([stats[i][0]['sum'] for i in range(len(stats))])
    all_sums_squared = sum([stats[i][0]['sum_squares'] for i in range(len(stats))])

    avg = all_sums / all_count
    var = ((all_sums_squared / all_count) - (avg ** 2)) ** 0.5
    results = {
        'avg': round(avg, 2),
        'std': round(np.sqrt(var), 2),
        'hist': {}
    }

    for key in ['1M', '10M', '100M', '1B', '1B+']:
        result = sum([stats[i][0][key] for i in range(len(stats))])
        results['hist'][key] = int(result)

    return results


def parallel_pipeline():
    files = os.listdir('files')
    mpi = MPI.COMM_WORLD
    world_size = mpi.Get_size()
    world_rank = mpi.Get_rank()
    num_files = len(files)

    min_step = num_files // world_size
    unallocated_files = num_files % world_size

    start_files = []
    end_files = []

    if world_rank == 0:

        start = 0

        if unallocated_files == 0:
            end = min_step
        else:
            end = min_step + 1
            unallocated_files -= 1

        for i in range(world_size):

            start_files.append(start)
            end_files.append(end)

            start = end
            if unallocated_files == 0:
                end = start + min_step
            else:
                unallocated_files -= 1
                end = start + min_step + 1

        start_files = np.array(start_files, dtype=np.int32)
        end_files = np.array(end_files, dtype=np.int32)

    else:
        end_files = None
        start_files = None

    start_local = np.empty(1, dtype=np.int32)
    end_local = np.empty(1, dtype=np.int32)

    mpi.Scatter(start_files, start_local, root=0)
    mpi.Scatter(end_files, end_local, root=0)

    start_file = start_local[0]
    end_file = end_local[0]


def main():
    start_time = time.time()
    args = parse_args()

    if args.mode == 'parallel':
        world_rank = MPI.COMM_WORLD.Get_rank()
        output = parallel_pipeline()

        if world_rank == 0:
            print(output)

            with open('output.json', 'w', encoding='utf8') as output_file:
                json.dump(output, output_file)

            end_time = time.time()
            print(round(end_time - start_time, 4))

    else:
        output = sequence_pipeline()
        print(output)

        with open('output.json', 'w', encoding='utf8') as output_file:
            json.dump(output, output_file)

    with open('output.json', 'w', encoding='utf8') as output_file:
        json.dump(output, output_file)


        end_time = time.time()
        print(round(end_time - start_time, 4))

if __name__ == '__main__':
    main()
