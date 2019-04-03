import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.upper = BaseBottleneck(in_planes, planes, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.upper(x)
#         print(out.shape)
        id_x = self.shortcut(x)
#         print(id_x.shape)
        out += id_x
        out = F.relu(out)
        return out
    
    def pass_through_grad(self, module, grad_in, grad_out):
        return grad_in

    def register_hooks_only_shortcut(self):
        self.upper_handle = self.upper.register_backward_hook(self.pass_through_grad)
    
    def register_hooks_only_block(self):
        self.shortcut_handle = self.shortcut.register_backward_hook(self.pass_through_grad)
        
    def unregister_hooks_only_shortcut(self):
        self.upper_handle.remove()
        
    def unregister_hooks_only_block(self):
        self.shortcut_handle.remove()
    
    
class BaseBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(BaseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out
    
    
class Base(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(Base, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )    
            
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.l2(out)
        return out
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.upper = Base(in_planes, planes, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
    def pass_through_grad(self, module, grad_in, grad_out):
        return grad_in

    def forward(self, x):
        out = self.upper(x)
        id_x = self.shortcut(x)
        out += id_x
        out = F.relu(out)
        return out
    
    def register_hooks_only_shortcut(self):
        self.upper_handle = self.upper.register_backward_hook(self.pass_through_grad)
    
    def register_hooks_only_block(self):
        self.shortcut_handle = self.shortcut.register_backward_hook(self.pass_through_grad)
        
    def unregister_hooks_only_shortcut(self):
        self.upper_handle.remove()
        
    def unregister_hooks_only_block(self):
        self.shortcut_handle.remove()


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.blocks = []
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        # Storage variables for various hooks measuring gradient magnitude
        self.input_grad_magnitude = []
        self.hook_dict = dict({})
        
        # Create path length dict
        self.path_length = 1
        self.path_length_magnitudes = dict({})
        for i in range(0,np.sum(num_blocks)+1):
            self.path_length_magnitudes[i] = []

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        layer = nn.Sequential(*layers)
        self.blocks.extend(layers)
#         layer.register_backward_hook(back_hook)
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def set_path_length(self, i):
        self.path_length = i
    
    def measure_input_grad(self, module, grad_in, grad_out):
        m = torch.norm(grad_out[0], p='fro').item()
        self.path_length_magnitudes[self.path_length].append(m)
        print('grad at input: {}'.format(m))
        
    def register_path_length_hooks(self):
        # Sample a self.path_length number of blocks
        indices = list(range(len(net.blocks)))
        selected_indices = np.random.choice(indices, size=(self.path_length), replace=False)
        unselected_indices = list(set(indices) - set(selected_indices))
        unselected_blocks = []
        for idx in unselected_indices:
            unselected_blocks.append(net.blocks[idx])

        selected_blocks = []
        for idx in selected_indices:
            selected_blocks.append(net.blocks[idx])
            
        self.selected_blocks = selected_blocks
        self.unselected_blocks = unselected_blocks
        
        for _block in self.selected_blocks:
            _block.register_hooks_only_block()
        for _block in self.unselected_blocks:
            _block.register_hooks_only_shortcut()
            
    def unregister_path_length_hooks(self):
        for _block in self.selected_blocks:
            _block.unregister_hooks_only_block()
        for _block in self.unselected_blocks:
            _block.unregister_hooks_only_shortcut()
        
    
    def register_hooks(self):
        input_grad_handle = self.conv1.register_backward_hook(self.measure_input_grad)
        self.hook_dict['input_grad'] = input_grad_handle
        self.register_path_length_hooks()
        
    def unregister_hooks(self):
        input_grad_handle = self.hook_dict['input_grad']
        input_grad_handle.remove()
        del self.hook_dict['input_grad']
        self.unregister_path_length_hooks()


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

if __name__ == '__main__': 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    magnitudes_dict = dict({})
    
    for pl in range(0,34):
        print('Measuring characteristic paths of length {}'.format(pl))
        trainset = torchvision.datasets.CIFAR10(root='./cifar-data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./cifar-data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net = ResNet101()
        net = net.to(device)
        net.set_path_length(pl)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        running_loss = 0.0
        train_loss = []
        val_loss = []
        PRINT = 10
        MEASURE_GRAD = 10
        N_EPOCH = 2

        for epoch in range(N_EPOCH):  # loop over the dataset multiple times
            print('Starting epoch {}'.format(epoch))
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Attach hooks to measure magnitude of input grad through a "characteristic path"
                if (i+1) % MEASURE_GRAD == 0:
                    net.register_hooks()
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Detach hooks for all layers
                if (i+1) % MEASURE_GRAD == 0:
                    net.unregister_hooks()

                # print statistics
                running_loss += loss.item()
                if (i+1) % PRINT == 0:    # print every PRINT mini-batches
                    train_loss.append(running_loss/PRINT)
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / PRINT))
                    running_loss = 0.0

        print('Finished Training')
        magnitudes_dict[pl] = net.path_length_magnitudes[pl]

    magnitudes_df = pd.DataFrame(magnitudes_dict)
    magnitudes_df.to_csv('grad_flow_experiment.csv', index=False)
    