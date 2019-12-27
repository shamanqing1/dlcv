"""Base dataset methods."""
import os
from torch.utils.data import Dataset

# pylint: disable= arguments-differ,unused-argument,missing-docstring,abstract-method

class ClassProperty(object):
    """Readonly @ClassProperty descriptor for internal usage."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class VisionDataset(Dataset):
    """Base Dataset with directory checker.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by default is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)


class KeyPointDataset(VisionDataset):
    """Base Dataset for KeyPoint detection.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by defaut is '~/.datasets/foo', where
        `foo` is the name of the dataset.
    """
    def __init__(self, root):
        super(KeyPointDataset, self).__init__(root)

    @property
    def num_joints(self):
        """Dataset defined: number of joints provided."""
        return 0

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return []

    @property
    def parent_joints(self):
        """A dict that defines joint id -> parent_joint_id mapping if applicable, can be empty."""
        return {}