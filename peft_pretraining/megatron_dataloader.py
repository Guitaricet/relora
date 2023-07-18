# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Tuple
from itertools import zip_longest
from functools import partial

import torch
import torch.utils.data
import torch.distributed as dist
import numpy as np

from loguru import logger

from peft_pretraining.megatron_indexed_dataset import make_dataset as make_indexed_dataset
from peft_pretraining.megatron_blendable_dataset import BlendableDataset
from peft_pretraining.megatron_dataset import GPT2Dataset


def make_data_loader(dataset, batch_size, num_workers):
    """Build dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    global_batch_size = batch_size * world_size

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
    )


def build_the_dataset(
    data_prefix,
    name,
    data_impl,
    num_samples,
    seq_length,
    seed,
    skip_warmup,
    build_index_mappings=True,
    label_prefix=None,
):
    """Build train/valid/test datasets."""

    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    if label_prefix is None:
        label_dataset = None
    else:
        label_dataset = make_indexed_dataset(label_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    logger.info("    {}:".format(name))
    logger.info("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    dataset = GPT2Dataset(
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=build_index_mappings,
        label_dataset=label_dataset,
    )
    return dataset


def build_train_valid_test_datasets(
    data_prefix,
    use_shared_fs,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logger.info(" > dataset split:")

    def print_split_stats(name, index):
        logger.info("    {}:".format(name))
        logger.info(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
            )

            dataset = GPT2Dataset(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                use_shared_fs=use_shared_fs,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return train_dataset, valid_dataset, test_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    return weights, weighted_num_samples


def build_weighted_datasets(
    neox_args,
    train_num_samples,
    valid_num_samples,
    test_num_samples,
    train_weights,
    valid_weights,
    test_weights,
    build_index_mappings=True,
):
    # build individual datasets
    train_datasets, valid_datasets, test_datasets = [], [], []
    for i, (train_path, label_path, valid_path, test_path) in enumerate(
        zip_longest(
            neox_args.train_data_paths,
            neox_args.label_data_paths if neox_args.label_data_paths else [],
            neox_args.valid_data_paths,
            neox_args.test_data_paths,
        )
    ):
        if train_path:
            train_datasets.append(
                build_the_dataset(
                    data_prefix=train_path,
                    name=f"train_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=train_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    label_prefix=label_path,
                )
            )

        if valid_path:
            valid_datasets.append(
                build_the_dataset(
                    data_prefix=valid_path,
                    name=f"valid_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=valid_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                )
            )

        if test_path:
            test_datasets.append(
                build_the_dataset(
                    data_prefix=test_path,
                    name=f"test_{i}",
                    data_impl=neox_args.data_impl,
                    num_samples=test_num_samples[i],
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                )
            )
    return train_datasets, valid_datasets, test_datasets


def weights_by_num_docs(l: list, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """
    if len(l) == 1:
        return [1.0]

    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i**alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights


def build_train_valid_test_data_iterators(neox_args):
    """XXX"""
    # Removed pipeline parallel from it to get rid of the mp

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    logger.info("> building train, validation, and test datasets ...")

    # Number of train/valid/test samples.
    train_iters = neox_args.train_iters
    eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
    test_iters = neox_args.eval_iters
    train_val_test_num_samples = [
        train_iters * neox_args.train_batch_size,
        eval_iters * neox_args.train_batch_size,
        test_iters * neox_args.train_batch_size,
    ]

    if neox_args.train_data_paths:
        # when individual train / valid / test data paths are provided
        # normalize weight values and get num samples for each dataset
        train_weights, train_num_samples = get_normalized_weights_and_num_samples(
            neox_args.train_data_weights, train_val_test_num_samples[0]
        )
        valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
            neox_args.valid_data_weights, train_val_test_num_samples[1]
        )
        test_weights, test_num_samples = get_normalized_weights_and_num_samples(
            neox_args.test_data_weights, train_val_test_num_samples[2]
        )

        # build individual datasets
        train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
            neox_args,
            train_num_samples,
            valid_num_samples,
            test_num_samples,
            train_weights,
            valid_weights,
            test_weights,
            build_index_mappings=not neox_args.weight_by_num_documents,
        )

        if neox_args.weight_by_num_documents:

            # gets the number of documents in each datapath
            get_num_docs_list = lambda datasets: [
                dataset.indexed_dataset.sizes.shape[0] for dataset in datasets
            ]
            train_num_docs, valid_num_docs, test_num_docs = (
                get_num_docs_list(train_datasets),
                get_num_docs_list(valid_datasets),
                get_num_docs_list(test_datasets),
            )

            # builds weights according to alpha + the number of docs
            fn = partial(
                weights_by_num_docs, alpha=neox_args.weighted_sampler_alpha
            )
            train_weights, valid_weights, test_weights = (
                fn(train_num_docs),
                fn(valid_num_docs),
                fn(test_num_docs),
            )
            (
                train_weights,
                train_num_samples,
            ) = get_normalized_weights_and_num_samples(
                train_weights, train_val_test_num_samples[0]
            )
            (
                valid_weights,
                valid_num_samples,
            ) = get_normalized_weights_and_num_samples(
                valid_weights, train_val_test_num_samples[1]
            )
            test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                test_weights, train_val_test_num_samples[2]
            )

            # rebuild datasets weighted according to new weights
            train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                neox_args,
                train_num_samples,
                valid_num_samples,
                test_num_samples,
                train_weights,
                valid_weights,
                test_weights,
            )

        if train_datasets:
            train_ds = BlendableDataset(train_datasets, train_weights)
        if valid_datasets:
            valid_ds = BlendableDataset(valid_datasets, valid_weights)
        if test_datasets:
            test_ds = BlendableDataset(test_datasets, test_weights)
    else:
        # when just data_path is provided
        # split dataset into train, valid and test from data_path
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=neox_args.data_path,
            use_shared_fs=neox_args.use_shared_fs,
            data_impl=neox_args.data_impl,
            splits_string=neox_args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=neox_args.seq_length,
            seed=neox_args.seed,
            skip_warmup=(not neox_args.mmap_warmup),
        )

    # Build dataloders.
    train_dataloader = make_data_loader(train_ds, neox_args.batch_size, neox_args.num_workers)
    valid_dataloader = make_data_loader(valid_ds, neox_args.batch_size, neox_args.num_workers)
    test_dataloader = make_data_loader(test_ds, neox_args.batch_size, neox_args.num_workers)

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and neox_args.train_iters > 0
    do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
    do_test = test_dataloader is not None and neox_args.eval_iters > 0
    # Need to broadcast num_tokens and num_type_tokens.
    flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])

    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (
            neox_args.iteration * neox_args.gradient_accumulation_steps
        ) % len(train_dataloader)
        logger.info(
            "setting training data start iteration to {}".format(
                train_dataloader.batch_sampler.start_iter
            )
        )
    if valid_dataloader is not None:
        start_iter_val = (
            (neox_args.iteration * neox_args.gradient_accumulation_steps)
            // neox_args.eval_interval
        ) * neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % len(
            valid_dataloader
        )
        logger.info(
            "setting validation data start iteration to {}".format(
                valid_dataloader.batch_sampler.start_iter
            )
        )

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


class DistributedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Similar to normal implementation of distributed sampler, except
    implementation is at the batch sampler level, instead of just the
    sampler level. This allows wrapping of arbitrary data samplers
    (sequential, random, WeightedRandomSampler, etc.) with this batch
    sampler.

    The `interleave` argument specifies how to distribute a batch. A value
    of True combined with the above random sampler is equivalent to pytorch's
    torch.utils.data.distributed.DistributedSampler.

    For the following batch [0,1,2,3,4,5,6,7] and data parallelism of 2
    specifying True will result in the following samples for each gpu:
        GPU0: [0,2,4,6] GPU1: [1,3,5,7]
    specifying False will result in the following samples:
        GPU0: [0,1,2,3] GPU1: [4,5,6,7]"""

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        rank=-1,
        world_size=2,
        wrap_last=False,
        interleave=False,
    ):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, "should not be here"
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.interleave = interleave

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter:
                    yield tbatch
                    self.start_iter = 0
                i += 1
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= self.batch_size
                self.wrap_around += len(batch)
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        if self.interleave:
            return batch[self.rank : self.batch_size : self.world_size]
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]
