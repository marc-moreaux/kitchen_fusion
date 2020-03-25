import torch
from torch import nn, optim
from torch.utils.data import dataloader
import sys
sys.path.insert(0, '../bc_learning_sound')
sys.path.insert(0, '../3D-ResNets-PyTorch')
sys.path.insert(0, '../audio')
sys.path.insert(0, '../GulpIO')

from bc_learning_sound.models import envnetv2_torch as EnvNetV2
from resnet3d_pytorch.models.resnet import ResNet
from dataloader import get_audio_data, get_video_data
# TESTER EPICVIDEODATASET ET LE BRANCHER SUR 3D-RESNET (win 2)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Load audio dataset
audios = get_audio_data()
n_nouns = len(audios['train_dataset'].nouns)
n_verbs = len(audios['train_dataset'].verbs)

# Load audio model
model = EnvNetV2(70)
model.load_chainer(
    ('/media/moreaux/a2b1d90b-93ea-4c5f-8adc-eef601253b99/results/'
     'envnet_results/esc70_ev2_strong_BC/1/model_split1.npz'))
model.fine_tune(n_nouns)
model.dual_output(n_verbs)
model.to(device)

# Set criterions
criterion_noun = nn.CrossEntropyLoss(
    audios['train_dataset'].noun_weights.float().to(device))
criterion_verb = nn.CrossEntropyLoss(
    audios['train_dataset'].verb_weights.float().to(device))
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)


# Train and test (valid)
for epoch in range(0):
    running_loss = .0
    total = 0
    correct = 0
    for i, (data, label, weight) in enumerate(audios['train_loader']):
        data = data.to(device)
        lbl_noun = label[0].to(device)
        lbl_verb = label[1].to(device)

        # Forward; backward; optimize
        optimizer.zero_grad()
        output_noun, output_verb = model(data)
        loss_noun = criterion_noun(output_noun, lbl_noun)
        loss_verb = criterion_verb(output_verb, lbl_verb)
        loss_noun.backward(retain_graph=True)
        loss_verb.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss_noun.item()
        if i % 30 == 29:    # print every 30 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0

        _, predicted = torch.max(output_noun.data, 1)
        total += lbl_noun.size(0)
        correct += (predicted == lbl_noun).sum().item()

    print('Accuracy of the network on the train dataset: %d %%' % (
          100 * correct / total))

    # Validation loop
    total = 0
    correct = 0
    for data, lbl in audios['valid_loader']:
        lbl_noun, lbl_verb = lbl
        data = data.to(device)
        lbl_noun, lbl_verb = lbl_noun.to(device), lbl_verb.to(device)

        output_noun, output_verb = model(data)
        _, predicted = torch.max(output_noun.data, 1)
        total += lbl_noun.size(0)
        correct += (predicted == lbl_noun).sum().item()

    print('Accuracy of the network on the valid dataset: %d %%' % (
          100 * correct / total))


################
# Video training

# Load video dataset
videos = get_video_data()

# Load video model
from resnet3d_pytorch import model, opts
opts = opts.parse_opts()
opts.root_path='/home/moreaux/work/3D-ResNets-PyTorch/data/'
opts.n_classes = n_nouns + n_verbs
opts.ft_begin_index = 4
print(opts)
model, model_params = model.generate_model(opts)

criterion_noun = nn.CrossEntropyLoss(
    audios['train_dataset'].noun_weights.float().to(device))
criterion_verb = nn.CrossEntropyLoss(
    audios['train_dataset'].verb_weights.float().to(device))
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)



# Train video model
for epoch in range(100):
    running_loss = .0
    total = 0
    correct = 0
    for i, (data, label, weight) in enumerate(videos['train_loader']):
        data = data.to(device)
        lbl_verb = label[0].to(device)
        lbl_noun = label[1].to(device)

        # Forward; backward; optimize
        optimizer.zero_grad()
        output = model(data)
        output_noun, output_verb = (output[:, :n_nouns], 
                                    output[:, n_nouns:])

        loss_noun = criterion_noun(output_noun, lbl_noun)
        #loss_verb = criterion_verb(output_verb, lbl_verb)
        loss_noun.backward(retain_graph=True)
        #loss_verb.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss_noun.item()
        if i % 30 == 29:    # print every 30 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0

        _, predicted = torch.max(output_noun.data, 1)
        total += lbl_noun.size(0)
        correct += (predicted == lbl_noun).sum().item()

    print('Accuracy of the network on the train dataset: %d %%' % (
          100 * correct / total))

    # Validation loop
    total = 0
    correct = 0
    for data, lbl in videos['valid_loader']:
        lbl_noun, lbl_verb = lbl
        data = data.to(device)
        lbl_noun, lbl_verb = lbl_noun.to(device), lbl_verb.to(device)

        output = model(data)
        output_noun, output_verb = (output[:, :n_nouns], 
                                    output[:, n_nouns:])
        _, predicted = torch.max(output_noun.data, 1)
        total += lbl_noun.size(0)
        correct += (predicted == lbl_noun).sum().item()

    print('Accuracy of the network on the valid dataset: %d %%' % (
          100 * correct / total))
