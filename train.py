from dataloader import EpicAudioDataset, EpicVideoDataset
from torch.utils.data import dataloader

import torchaudio.transforms as audio_transforms
import torchvision.transforms as video_transforms

import bc_learning_sound
from bc_learning_sound.models.envnetv2_torch import EnvNetV2

# TESTER EPICVIDEODATASET ET LE BRANCHER SUR 3D-RESNET (win 2)

input_length = 4100
audio_train_transform = audio_transforms.Compose([
    audio_transforms.ToTensor(),
    audio_transforms.RandomStretch(1.25),
    audio_transforms.Scale(2 ** 16 / 2),
    audio_transforms.Pad(input_length // 2),
    audio_transforms.RandomCrop(input_length),
    audio_transforms.RandomOpposite()])


video_train_transform = video_transforms.Compose([
    video_transforms.RandomResizedCrop(224),
    video_transforms.RandomGrayscale(),
    video_transforms.RandomVerticalFlip(),
    video_transforms.ToTensor(),
    video_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


audio_dataset = EpicAudioDataset(
    '../starter-kit-action-recognition/data/processed/gulp/audio_train',
    transform=audio_train_transform,
    get_output_weighting=True,
    samples_per_verb=10)


audio_train_loader = dataloader.DataLoader(
    audio_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=0,
    drop_last=True)

# Load audio model
model = EnvNetV2(70)
model.load_chainer('/media/moreaux/a2b1d90b-93ea-4c5f-8adc-eef601253b99/results/envnet_results/esc70_ev2_strong_BC/1/model_split1.npz')

for data, label, weight in audio_train_loader:
    print(data, label, weight)
    break


# video_dataset = EpicVideoDataset(
    # '../starter-kit-action-recognition/data/processed/audio_train',
    # num_frames=16,
    # step_size=1,
    # is_val=False,
    # transform=video_train_transform,
    # random_offset=True)
