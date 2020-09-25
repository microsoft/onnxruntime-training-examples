# Copyright (c) 2020 Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import concurrent.futures
import logging
import os
import random

import h5py
import numpy
import torch

from . import configuration

class BertMultiFileDataset(torch.utils.data.IterableDataset):

    def __init__(self, files, loop=False):        
        super().__init__()
        self.files = files
        self.loop = loop
        random.shuffle(self.files)

        self.file_index = 0
        self.sample_index = 0
        self.pool = None
        self.dataset = None
        self.future_dataset = None

    def forward(self, count):
        for idx, _ in enumerate(self):
            if count <= idx:
                break
        logging.debug('Forwarded dataset to file_index {}, sample_index {}'.format(
            self.file_index, self.sample_index))
        self._destroy_datasets_if_exist()
        self._destroy_threadworker_if_exists()

    def reset(self):
        self.file_index = 0
        self.sample_index = 0
        self._destroy_datasets_if_exist()
        self._destroy_threadworker_if_exists()

    # note: iterator invoked by worker process (not the contructing process)
    # note: state of iterator resumes from (self.file_index, self.sample_index)
    def __iter__(self):
        self._create_threadworker_if_not_exists()
        self._fetch_future_dataset_or_none(self.file_index)

        while self.future_dataset is not None:
            self.dataset = self.future_dataset.result(timeout=None)
            self._fetch_future_dataset_or_none(self.file_index + 1)

            while self.sample_index < len(self.dataset):
                yield self.dataset[self.sample_index]
                self.sample_index += 1

            self.file_index += 1
            self.sample_index = 0

        self.reset()

    def _create_threadworker_if_not_exists(self):
        if self.pool is None:
            self.pool = concurrent.futures.ThreadPoolExecutor(1)            

    def _fetch_future_dataset_or_none(self, index):
        dataset_factory = lambda filepath: BertSingleFileDataset(filepath)
        if self.loop or index < len(self.files):
            next_file = self.files[index % len(self.files)]
            self.future_dataset = self.pool.submit(dataset_factory, next_file)
        else:
            self.future_dataset = None

    def _destroy_threadworker_if_exists(self):
        if self.pool is not None:
            del self.pool
        self.pool = None
    
    def _destroy_datasets_if_exist(self):
        if self.dataset is not None:
            del self.dataset
        self.dataset = None
        if self.future_dataset is not None:
            del self.future_dataset
        self.future_dataset = None

class BertSingleFileDataset(torch.utils.data.Dataset):

    def __init__(self, hdf5_filepath):
        super().__init__()

        # refer bert_model.py for description of input meanings
        self.input_names = [
            'input_ids',
            'segment_ids',
            'input_mask',
            'masked_lm_positions',
            'masked_lm_ids',
            'next_sentence_labels'
        ]

        # load file data into memory as numpy arrays
        # len(self.bulk_data['input_ids]') = <number of samples>
        self.bulk_data = {}
        with h5py.File(hdf5_filepath, 'r') as hdf5_data:
            for name in self.input_names:
                self.bulk_data[name] = numpy.asarray(hdf5_data[name][:])

        logging.debug('Loaded {} samples from {}'.format(
            len(self.bulk_data['input_ids']), hdf5_filepath))

    def __len__(self):
        return len(self.bulk_data['input_ids'])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(
                numpy.asarray(
                    self.bulk_data[name][index], dtype=numpy.int64)
                ) for name in self.input_names]

        masked_lm_labels = self._build_masked_lm_labels(masked_lm_positions, masked_lm_ids)
        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]

    # construct masked_lm_labels from masked_lm_positions and masked_lm_ids
    #   correct label at ith position if ith input token is [MASK] (and -1 otherwise)
    def _build_masked_lm_labels(self, masked_lm_positions, masked_lm_ids):
        masked_token_count = self._get_masked_token_count(masked_lm_positions)

        masked_lm_labels = torch.ones([configuration.arguments.max_seq_length], dtype=torch.int64) * -1
        masked_lm_labels[masked_lm_positions[:masked_token_count]] = masked_lm_ids[:masked_token_count]
        return masked_lm_labels

    def _get_masked_token_count(self, masked_lm_positions):
        masked_token_count = configuration.arguments.max_predictions_per_seq
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            masked_token_count = padded_mask_indices[0].item()
        return masked_token_count