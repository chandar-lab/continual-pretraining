import os
import sys

import importlib
import inspect
from argparse import Namespace

# from continual_dataset import ContinualDataset
# from conf import warn_once


def get_all_datasets():
    """Returns the list of all the available datasets in the datasets folder."""
    return [model.split('.')[0] for model in os.listdir('data')
            if not model.find('__') > -1 and 'py' in model]


# def get_dataset(args: Namespace) -> ContinualDataset:
#     """
#     Creates and returns a continual dataset among those that are available.
#     If an error was detected while loading the available datasets, it raises the appropriate error message.

#     Args:
#         args (Namespace): the arguments which contains the hyperparameters

#     Exceptions:
#         AssertError: if the dataset is not available
#         Exception: if an error is detected in the dataset

#     Returns:
#         the continual dataset instance
#     """
#     names = get_dataset_names()
#     assert args.dataset in names
#     return get_dataset_class(args)(args)


# def get_dataset_class(args: Namespace) -> ContinualDataset:
#     """
#     Return the class of the selected continual dataset among those that are available.
#     If an error was detected while loading the available datasets, it raises the appropriate error message.

#     Args:
#         args (Namespace): the arguments which contains the `--dataset` attribute

#     Exceptions:
#         AssertError: if the dataset is not available
#         Exception: if an error is detected in the dataset

#     Returns:
#         the continual dataset class
#     """
#     names = get_dataset_names()
#     assert args.dataset in names
#     if isinstance(names[args.dataset], Exception):
#         raise names[args.dataset]
#     return names[args.dataset]


# def get_dataset_names():
    # """
    # Return the names of the selected continual dataset among those that are available.
    # If an error was detected while loading the available datasets, it raises the appropriate error message.

    # Args:
    #     args (Namespace): the arguments which contains the `--dataset` attribute

    # Exceptions:
    #     AssertError: if the dataset is not available
    #     Exception: if an error is detected in the dataset

    # Returns:
    #     the continual dataset class names
    # """

    # def _dataset_names():
    #     names = {}
    #     for dataset in get_all_datasets():
    #         try:
    #             mod = importlib.import_module('data.' + dataset)
    #             dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
    #                                     and 'ContinualDataset' in str(inspect.getmro(getattr(mod, x))[1:]) and 'GCLDataset' not in str(inspect.getmro(getattr(mod, x)))]
    #             for d in dataset_classes_name:
    #                 c = getattr(mod, d)
    #                 names[c.NAME] = c

    #             gcl_dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    #             for d in gcl_dataset_classes_name:
    #                 c = getattr(mod, d)
    #                 names[c.NAME] = c
    #         except Exception as e:
    #             warn_once(f'Error in dataset {dataset}')
    #             warn_once(e)
    #             names[dataset.replace('_', '-')] = e
    #     return names

    # if not hasattr(get_dataset_names, 'names'):
    #     setattr(get_dataset_names, 'names', _dataset_names())
    # return getattr(get_dataset_names, 'names')
