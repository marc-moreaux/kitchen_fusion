from gulpio import GulpDirectory
from gulpio.dataset import GulpIOEmptyFolder
from PIL import Image
import numpy as np
import pandas as pd
import random
import torch
import collections
from torch.utils.data import dataloader
import torchaudio.transforms as audio_transforms
import torchvision.transforms as video_transforms


class EpicAudioDataset(object):

    def __init__(self, data_path,
                 is_val=False,
                 transform=None,
                 get_output_weighting=False,
                 samples_per_verb=-1):
        r"""Simple data loader for GulpIO format.

            Args:
                data_path (str): path to GulpIO dataset folder
                is_val (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                get_output_weighting (bool): yield the class weigthing when
            iterating.
                samples_per_verb (int): amount of samples per verb-class
        """

        # Load the underlying dataset
        self.data_path = data_path
        self.gd = GulpDirectory(data_path, encode_jpg=False)
        self._items = list(self.gd.merged_meta_dict.items())
        self.items = sorted([i[1]['meta_data'][0] for i in self._items],
                            key=lambda x: x['uid'])
        self.num_chunks = self.gd.num_chunks

        # Copy parameters
        self.get_output_weighting = get_output_weighting
        self.transform = transform
        self.is_val = is_val

        # Get the class names
        nouns = pd.read_csv('./EPIC_noun_classes.csv')
        verbs = pd.read_csv('./EPIC_verb_classes.csv')
        self.nouns = list(nouns['nouns'])
        self.verbs = list(verbs['verbs'])
        self.classes = self.verbs.copy()
        self.classes.append(self.verbs)
        self.n_noun_classes = len(nouns)
        self.n_verb_classes = len(verbs)

        # Select the desired amount of samples per verb-class
        self.samples_per_verb = samples_per_verb
        self._select_verb_samples()

        # Set the class weighting
        if get_output_weighting:
            self._set_weigthing()

        # Count the amout of samples per class
        self.verb_counter = collections.Counter(
            [meta['verb'] for meta in self.items])
        self.noun_counter = collections.Counter(
            [meta['noun'] for meta in self.items])

        if self.num_chunks == 0:
            raise(GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                    "of: ".format(data_path)))

        print(" > Found {} chunks".format(self.num_chunks))

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        # Get audio
        gd_id = self.items[index]['uid']
        audio, _ = self.gd[gd_id]
        audio = audio[0]
        if audio.dtype != np.dtype('int16'):
            audio = np.frombuffer(audio.tostring(), np.int16)
        if audio.ndim != 2:
            audio = audio.reshape(2, -1)

        # Augmentation
        if self.transform:
            audio = self.transform(audio)

        # Labels
        noun_target_idx = self.items[index]['noun_class']
        verb_target_idx = self.items[index]['verb_class']
        labels = verb_target_idx, noun_target_idx

        # Weighting
        if self.get_output_weighting:
            weighting = (self.verb_weights[verb_target_idx],
                         self.noun_weights[noun_target_idx])
            return audio, labels, weighting
        else:
            return audio, labels

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.items)

    def _set_weigthing(self):
        """
        Computes the class weightning for the noun and the verbs based on the
        amount of samples per class
        """
        nouns = np.array([0.] * self.n_noun_classes)
        verbs = np.array([0.] * self.n_verb_classes)

        for meta in self.items:
            nouns[meta['noun_class']] += 1
            verbs[meta['verb_class']] += 1

        self.noun_weights = torch.tensor(nouns / nouns.sum())
        self.verb_weights = torch.tensor(verbs / verbs.sum())

    def _select_verb_samples(self):
        """Select N samples per verb-class (at max)
        """
        if self.samples_per_verb <= 0:
            return

        items_keep = []
        count_track = [0] * self.n_verb_classes
        for meta in sorted(self.items,
                           key=lambda x: x['verb_class']):
            if count_track[meta['verb_class']] < self.samples_per_verb:
                count_track[meta['verb_class']] += 1
                if not self.is_val:
                    items_keep.append(meta)
            else:
                if self.is_val:
                    items_keep.append(meta)
        self.items = items_keep

    def __repr__(self):
        tmp = '{} elements\n'.format(len(self.items))
        tmp = collections.Counter([meta['verb'] for meta in self.items])
        tmp = str(tmp)[9:-2].replace("'", '')
        _str = tmp + '\n'
        tmp = collections.Counter([meta['noun'] for meta in self.items])
        tmp = str(tmp)[9:-2].replace("'", '')
        _str += tmp
        return _str

    def __str__(self):
        return ('AudioDataset with {} samples, {} verb-classes, {}/{} ' +
                'samples per verb class, {}/{} samples per noun class').format(
            len(self.items),
            self.n_verb_classes,
            min(self.verb_counter.values()), max(self.verb_counter.values()),
            min(self.noun_counter.values()), max(self.noun_counter.values()))


class EpicVideoDataset(EpicAudioDataset):

    def __init__(self, data_path, num_frames, step_size,
                 is_val, transform=None, target_transform=None, stack=True,
                 random_offset=True,
                 get_output_weighting=False,
                 samples_per_verb=-1):
        r"""Simple data loader for GulpIO format.

            Args:
                data_path (str): path to GulpIO dataset folder
                num_frames (int): number of frames to be fetched.
                step_size (int): number of frames skippid while picking
            sequence of frames from each video.
                is_val (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                target_transform (func):  a transformation function applied to each
            single target, where target is the id assigned to a label. The
            mapping from label to id is provided in the `label_idx` member-
            variable. Default is None.
                stack (bool): stack frames into a numpy.array. Default is True.
                random_offset (bool): random offsetting to pick frames, if
            number of frames are more than what is necessary.
                get_output_weighting (bool): yield the class weigthing when
            iterating.
                samples_per_verb (int): amount of samples per verb-class
        """

        # Load the underlying dataset
        self.data_path = data_path
        self.gd = GulpDirectory(data_path)
        self._items = list(self.gd.merged_meta_dict.items())
        self.items = sorted([i[1]['meta_data'][0] for i in self._items],
                            key=lambda x: x['uid'])
        self.num_chunks = self.gd.num_chunks

        # Copy parameters
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.step_size = step_size
        self.is_val = is_val
        self.stack = stack
        self.random_offset = random_offset

        # Get the class names
        nouns = pd.read_csv('./EPIC_noun_classes.csv')
        verbs = pd.read_csv('./EPIC_verb_classes.csv')
        self.nouns = list(nouns['nouns'])
        self.verbs = list(verbs['verbs'])
        self.classes = self.verbs.copy()
        self.classes.append(self.verbs)
        self.n_noun_classes = len(nouns)
        self.n_verb_classes = len(verbs)

        # Select the desired amount of samples per verb-class
        self.samples_per_verb = samples_per_verb
        self._select_verb_samples()

        # Set the class weighting
        self._set_weigthing()
        self.get_output_weighting = get_output_weighting

        # Count the amout of samples per class
        if not is_val:
            self.verb_counter = collections.Counter(
                [meta['verb'] for meta in self.items])
            self.noun_counter = collections.Counter(
                [meta['noun'] for meta in self.items])

        if self.num_chunks == 0:
            raise(GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                    "of: ".format(data_path)))

        print(" > Found {} chunks".format(self.num_chunks))

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        # Get Frames and info
        meta = self.items[index]
        gd_id = meta['uid']
        frames, item_info = self.gd[gd_id]
        num_frames = len(frames)

        # set number of necessary frames
        if self.num_frames > -1:
            num_frames_necessary = self.num_frames * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames and self.random_offset:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        # set target frames to be loaded
        frames_slice = slice(offset, num_frames_necessary + offset,
                             self.step_size)
        frames, meta = self.gd[gd_id, frames_slice]

        # padding last frame
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frames.extend([frames[-1]] * (num_frames_necessary - num_frames))

        # Data augmentation per frame
        seed = np.random.randint(2147483647)
        random.seed(seed)  # apply this seed to img transforms
        if self.transform is not None:
            old_frames = frames
            frames = []
            for frame in old_frames:
                frame = Image.fromarray(frame)
                frames.append(self.transform(frame))

        # Data augmentation per class
        seed = np.random.randint(2147483647)
        random.seed(seed) # apply this seed to target transforms
        if self.target_transform is not None:
            old_frames = frames
            frames = []
            for frame in old_frames:
                frame = Image.fromarray(frame)
                frames.append(self.target_transform(frame))

        # format data to torch tensor
        if self.stack:
            frames = torch.stack(frames, -3)

        # Labels
        verb_target_idx = self.items[index]['verb_class']
        noun_target_idx = self.items[index]['noun_class']
        labels = verb_target_idx, noun_target_idx

        # Weighting
        if self.get_output_weighting:
            weighting = (self.verb_weights[verb_target_idx],
                         self.noun_weights[noun_target_idx])
            return frames, labels, weighting
        else:
            return frames, labels

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.items)


def get_audio_data(input_length=66650,
                   gulp_path=('../starter-kit-action-recognition/'
                              'data/processed/gulp/')):
    '''Create train and valid audio datasets
    '''
    train_transform = audio_transforms.Compose([
        audio_transforms.ToTensor(),
        audio_transforms.StereoToMono(),
        audio_transforms.RandomStretch(1.25),
        audio_transforms.Scale(2 ** 16 / 2),
        audio_transforms.Pad(input_length // 2),
        audio_transforms.RandomCrop(input_length),
        audio_transforms.RandomOpposite(),
        audio_transforms.AddDimension(1)])

    valid_transform = audio_transforms.Compose([
        audio_transforms.ToTensor(),
        audio_transforms.StereoToMono(),
        audio_transforms.Scale(2 ** 16 / 2),
        audio_transforms.Pad(input_length // 2),
        audio_transforms.RandomCrop(input_length),
        audio_transforms.AddDimension(1)])

    train_dataset = EpicAudioDataset(
        gulp_path + '/audio_train',
        transform=train_transform,
        get_output_weighting=True,
        samples_per_verb=100)

    valid_dataset = EpicAudioDataset(
        gulp_path + '/audio_train',
        transform=valid_transform,
        is_val=True,
        samples_per_verb=100)

    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    valid_loader = dataloader.DataLoader(
        valid_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    audios = {'train_dataset': train_dataset,
              'train_loader':train_loader,
              'valid_dataset':valid_dataset,
              'valid_loader':valid_loader}

    return audios


def get_video_data(gulp_path=('/media/moreaux/'
                              '82de644c-4c83-4f50-a60d-177c45516105/datasets/'
                              'Epic_kitchens/3h91syskeag572hl6tvuovwv4d/data/'
                              'processed/gulp/')):
    '''Create train and valid video datasets
    '''
    train_transform = video_transforms.Compose([
        video_transforms.RandomResizedCrop(112),
        video_transforms.RandomGrayscale(),
        video_transforms.RandomVerticalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225]),
    ])

    valid_transform = video_transforms.Compose([
        video_transforms.RandomResizedCrop(112),
        video_transforms.ToTensor(),
        video_transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225]),
    ])

    train_dataset = EpicVideoDataset(
        gulp_path + 'rgb_train',
        num_frames=16,
        step_size=1,
        is_val=False,
        transform=train_transform,
        get_output_weighting=True,
        samples_per_verb=100)

    valid_dataset = EpicVideoDataset(
        gulp_path + 'rgb_train',
        num_frames=16,
        step_size=1,
        is_val=True,
        transform=valid_transform,
        samples_per_verb=100)

    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=15,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    valid_loader = dataloader.DataLoader(
        valid_dataset,
        batch_size=20,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    videos = {'train_dataset': train_dataset,
              'train_loader': train_loader,
              'valid_dataset': valid_dataset,
              'valid_loader': valid_loader}

    return videos


if __name__ == '__main__':
    gulp_path = '../starter-kit-action-recognition/data/processed/gulp/'

    # Test audio
    input_length = 44100
    audio_transform = audio_transforms.Compose([
        audio_transforms.ToTensor(),
        audio_transforms.RandomStretch(1.25),
        audio_transforms.Scale(2 ** 16 / 2),
        audio_transforms.Pad(input_length // 2),
        audio_transforms.RandomCrop(input_length),
        audio_transforms.RandomOpposite()])

    dataset = EpicAudioDataset(
        gulp_path + 'audio_train',
        transform=audio_transform,
        get_output_weighting=True,
        samples_per_verb=15)

    train_loader = dataloader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    for data, label, weight in train_loader:
        print(data, label, weight)
        break

    # Test video
    video_transform = video_transforms.Compose([
        video_transforms.RandomResizedCrop(224),
        video_transforms.RandomGrayscale(),
        video_transforms.RandomVerticalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
    ])

    dataset = EpicVideoDataset(
        gulp_path + 'rgb_test_seen',
        num_frames=16,
        step_size=1,
        is_val=False,
        transform=video_transform,
        get_output_weighting=True,
        samples_per_verb=15)
    print(dataset)

    train_loader = dataloader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True)
