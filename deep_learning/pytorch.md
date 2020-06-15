<a name="top"></a>
## Table of Contents
<ul>
	
	<li><a href="#setup">Setup</a></li>
	<li><a href="#cuda">Cuda</a></li>
<li><a href="#tensor">Tensor</a></li>
	<li><a href="#network">Network</a></li>
	<li><a href="#features">Features</a></li>
      
</ul>

<a name="setup"></a>
### Setup
``` python
torch.cuda.is_available()
torch.version.cuda
```

<a name="Cuda"></a>
### Cuda
``` python

```

<a name="tensor"></a>
### Tensor

- Data that can be used by pytorch

``` python
t = torch.Tensor()
t.dtype
t.device
t.layout

device = torch.device('cuda:0')//first GPU
# tensor operations between tensors must happen on the same device.
data = np.array([1,2,3])

torch.Tensor(data) #default float32, global default
torch.get_default_dtype()

torch.tensor(data) #default datatype as original data

# The followings use the reference of the numpy array
torch.as_tensor(data) #same as above, accept datatype other than numpy array
torch.from_numpy(data) #same as above

torch.eye(2) # # of rows 2, identity
torch.zeros(2, 2)
torch.ones(2, 2)
torch.rand(2, 2)

t.reshape(3, 3)
t.shape

# tell how many elements
torch.tensor(t.shape).prod()
t.numel()

t.reshape(1, 12)
t.reshape(1, 12).squeeze() # 12, remove the length 1 dimension
t.reshape(1, 12).squeeze().unsqueeze(dim=0) # add new dimension at 0
t.reshape(1, -1) # -1 tells pytorch to figure out the second dimension by itself
torch.cat((t1, t2), dim=0)
torch.stack((t1, t2, t3)) #stack and insert axis 0 as m, then hight and width
t.reshape(-1) # To a 1 dimensional array
t.flatten(start_dim=1) # flat along 1 axis

# Element-wise operation
t1 + t2
t1 + 2
t1*2
t1/2
t1.add(2)
t1.sub(2)
t1.mul(2)
t1.eq(0)
t1.ge(0)
t1.gt(0)
t1.lt(0)
t1.le(0)

t.abs()
t.sqrt()
t.neg()


# broadcast automatically

np.broadcast_to(t2.numpy(), t1.shape)

#Reduction and ArgMax
t.sum(dim=0) # sum along axis 0
t.prod()
t.mean()
t.std()

t.mean().item() #get the value instead of a tensor
t.mean(dim=0).tolist()
t.mean(dim=0).numpy()

# if t 3 * 4
# [[1, 0, 0, 2],
	[0, 0, 0, 3],
	[4, 0, 0, 5]]
t.argmax() # result: 11
t.argmax(dim=0) # result: [2, 0, 0, 2]
#used to deal with the output to find the classes it predicted
```

<a name="network"></a>
### Network
- Fashion-mnist
	- replace the mnist
	- 10 classes
	- 28 * 28
	- 60000 training examples
	- 10000 testing examples

- Dataset, DataLoader
	- We need to implement a subclass of Dataset to use costumized dataset, then use the loader to load data

``` python
class OHLC(Dataset):
	def __init__(self, csv_file):
		self.data = pd.read_csv(csv_file)
		
	def __getitem__(self, index):
		r = self.data.iloc[index]
		label = torch.tensor(r.is_up_day, dtype=torch.long)
		sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
	
	def __len__(self):
		return len(self.data)
```

``` python
import torch
import torchvision
import torchvision.transforms as transforms

# multiple transform can be used to transform the data differently
train_set = torchvision.datasets.FashionMNIST(
	root='./data/FashionMNIST',
	train=True,
	download=True,
	transform=transforms.Compose([
		transforms.ToTensor() # just transform to a tensor
	])
)

train_loader = torch.utils.data.DataLoader(
	train_set, batch_size=10
)
len(train_set)
60000

train_set.train_labels

train_set.train_labels.bincount() # Gives the frequency of the values inside the tensor

sample = next(iter(train_set)) # return data, label

image, label = sample
image.shape # 1 * 28 * 28 channel * 28 * 28

batch = next(iter(train_loader))
images, labels = batch # images: 10, 1, 28, 28

# show images
grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))

print('labels:', labels)

```

- model example
	- Network or Layer class should extend torch.nn.Module
	- implement forward method
	- define layers in the constructor

``` python
class Network:
	def __init__(self):
		self.layer = None
		
	def forward(self, t):
		t = self.layer(t)
		return t
		
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__() 
		self.layer = None
		# kernel_size = (5, 5)
		# stride=(1, 1)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
		# bias=True
		self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
		self.fc2 = nn.Linear(in_features=120, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)
	
	def forward(self, t):
		# (1) input layer
		t = t
		
		# (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		
		# (3) hidden conv Layer
		t = self.conv2(t)
		t = F.relu(t) # no weights to learn using function in Functional module
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		
		# (4) hidden Linear layer
		t = t.reshape(-1, 12 * 4 * 4 )
		t = self.fc1(t)
		t = F.relu(t)
		
		# (5) hidden Linear Layer
		t = self.fc2(t)
		t = F.relu(t)
		
		# (6) output layer
		t = self.out(t)
		# t = F.softmax(t, dim=1)
		return t


torch.set_grad_enabled(True) # turn off automatically calculated gradients can save memory on computational graph but no automatically gradient (default on)
network = Network()
train_loader = torch.utils.data.DataLoader(tran_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)



for epoch in range(5):
	total_loss = 0
	total_correct = 0
	for batch in train_loader:
	#batch = next(iter(train_loader))
		images, labels = batch

#sample = next(iter(train_set))
#image, label = sample
#image.unsqueeze(0) # get a single exampe batch

# here we need a for loop for go through all the batches and epoches
		preds = network(image)
		loss = F.cross_entropy(preds, labels) #calculating the loss, it use softmax as default
		loss.item() # check loss value
		optimizer.zero_grad() # it add all the new grads by default, now we clear them
		loss.backward() #calculate all the parameters


		optimizer.step() # update the parameters
		total_loss += loss.item()
		total_correct += get_num_correct(preds, labels)



#F.softmax(pred, dim=1)

#pred.argmax(dim=1)
#pred.argmax(dim=1).eq(labels) # compare, labels is the position of the correct 
#F.softmax(pred, dim=1).sum() # should be 1



#kernel: (filter)
# print(network) show network
```

- Learnable
	- extend Parameter class
	- weight tensor

``` python
# filter index, channel, height, width for convolve layer
network.conv1.weight # default created

# calculation
weight_matrix = torch.tensor([
	[1, 2, 3, 4],
	[2, 3, 4, 5],
	[3, 4, 5, 6]
], dtype=torch.float32)

weight_matrix.matmul(in_features)

# print all parameters
for [name,]param in networ.parameters():
	print(param.shape)
	
# callable
# weight are initialized randomly
fc = nn.Linear(in_features=4, out_features=3)
# __call__(in) is implemented
fc(in_features)
# Don't call the forward method
# assign w
fc.weight = nn.Parameter(weight_matrix)


```

- build confusion matrix (prediction classes confuse the network)

``` python
#target_set.targets # labels

@torch.no_grad() # disable tracking on this function
def get_all_preds(model, loader):
	all_preds = torch.tensor([])
	for batch in loader:
		images, labels = batch
		preds = model(images)
		all_preds = torch.cat(
			(all_preds, preds)
			, dim=0
		)
	return all_preds
	
prediction_loader = torch.utils.data.Dataloader(train_set, batch_size=10000)
train_preds = get_all_preds(network, prediction_loader)

print(train_preds.requires_grad)

# train_preds.requires_grad check if auto grad is on or off
# train_preds.grad check grads
# train_preds.grad_fn check grad_fn

# use context to ensure any tracking is down without gradients, turn it off locally
with torch.no_grad():
	prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
	train_preds = get_all_preds(network, prediction_loader)
	
"""
Build confusion matrix
"""
train_set.targets
train_preds.argmax(dim=1)

stacked = torch.stack(
	(
		train_set.targets,
		train_preds.argmax(dim=1)
	)
)

stacked[0].tolist() # list of tuples

cmt = torch.zeros(10, 10, dtype=torch.int32)

for p in stacked:
	y, y_hat = p.tolist() # y, y_hat are the positions
	cmt[y, y_hat] = cmt[y, y_hat] + 1

cmt #show the result

# plot
import matplotlib.pyplot as plt

from resources.plotcm import plot_confusion_matrix

names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cmt, names)

```

- plot 


``` python
#resources.plotcm
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

- TensorBoard
	- web front end reads data and show in gui
	- SummaryWriter

``` shell
tensorboard --version
tensorboard --logdir=runs
# browse to localhost 6006
```


``` python
from torch.utils.tensorboard import SummaryWriter

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True) # can shuffle the data

comment = f' batch_size={batch_size} lr={lr}'
tb = SummaryWriter(comment=comment)
network = Network()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)
tb.add_graph(network, images)
tb.close()

# ... run the network

# add these after batches in every epoch are finished
tb.add_scalar('Loss', total_loss, epoch)
tb.add_scalar('Number Correct', total_correct, epoch)
tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

for name, weight in network.named_parameters():
	tb.add_histogram(name, weight, epoch)
	tb.add_histogram(f'{name}.grad', weight.grad, epoch)
	
tb.close()



# runs folder will have data

```

- hyperparameter tunning

``` python
paramters = dict(
	lr = [.01, .001],
	batch_size = [10, 100, 1000],
	shuffle = [True, False]
)

param_values = [v for v in parameters.values()] # a list contain 3 lists

# use this as outer loop
for lr, batch_size, shuffle in product(*param_values):
	print (lr, batch_size, shuffle)
	
# 0.01 10 True
# 0.01 10 False
# ...

```

- better way for the code

``` python
from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
        
params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
)

"""
> runs = RunBuilder.get_runs(params)
> runs

[
    Run(lr=0.01, batch_size=1000),
    Run(lr=0.01, batch_size=10000),
    Run(lr=0.001, batch_size=1000),
    Run(lr=0.001, batch_size=10000)
]"""

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
    ,device = ["cuda", "cpu"]
)

runs = RunBuilder.get_runs(params)

"""
Create runs
"""
Run = namedtuple('Run', params.keys())
runs = []
for v in product(*params.values()):
    runs.append(Run(*v))


"""
New outer loop
"""
for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
    
    # Training process given the set of parameters

```

- Evern better RunManager

``` python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None 

class RunManager():
	def __init__(self):

    	self.epoch_count = 0
    	self.epoch_loss = 0
    	self.epoch_num_correct = 0
    	self.epoch_start_time = None

    	self.run_params = None
    	self.run_count = 0
    	self.run_data = []
    	self.run_start_time = None

    	self.network = None
    	self.loader = None
    	self.tb = None
    	
   def begin_run(self, run, network, loader):

    	self.run_start_time = time.time()

    	self.run_params = run
    	self.run_count += 1

    	self.network = network
    	self.loader = loader
    	self.tb = SummaryWriter(comment=f'-{run}')

    	images, labels = next(iter(self.loader))
    	grid = torchvision.utils.make_grid(images)

    	self.tb.add_image('images', grid)
    	self.tb.add_graph(self.network, images)
    
    def end_run(self):
    	self.tb.close()
    	self.epoch_count = 0
    	
    def begin_epoch(self):
    	self.epoch_start_time = time.time()

    	self.epoch_count += 1
    	self.epoch_loss = 0
    	self.epoch_num_correct = 0
    	
    def end_epoch(self):

    	epoch_duration = time.time() - self.epoch_start_time
    	run_duration = time.time() - self.run_start_time

    	loss = self.epoch_loss / len(self.loader.dataset)
    	accuracy = self.epoch_num_correct / len(self.loader.dataset)

    	self.tb.add_scalar('Loss', loss, self.epoch_count)
    	self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

    	for name, param in self.network.named_parameters():
    		self.tb.add_histogram(name, param, self.epoch_count)
    		self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

    	results = OrderedDict()
    	results["run"] = self.run_count
    	results["epoch"] = self.epoch_count
    	results['loss'] = loss
    	results["accuracy"] = accuracy
    	results['epoch duration'] = epoch_duration
    	results['run duration'] = run_duration
    	for k,v in self.run_params._asdict().items(): 				
    		results[k] = v
    		self.run_data.append(results)
    ...
    
    def track_loss(self, loss):
    	self.epoch_loss += loss.item() * self.loader.batch_size

	def track_num_correct(self, preds, labels):
    	self.epoch_num_correct += self.get_num_correct(preds, labels)
    
    def _get_num_correct(self, preds, labels):
    	return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):
    
    	pd.DataFrame.from_dict(
        	self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

    	with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
        	json.dump(self.run_data, f, ensure_ascii=False, indent=4)
```
- training 


- speed up the training by use run.num_workers (speed up the reading data process)

### Final code

``` python
params = OrderedDict(
	lr = [.01],
	batch_size = [100, 1000, 10000],
	num_workers = [0, 1, 2, 4, 8, 16],
	shuffle = [True, False]
)
m = RunManager()
for run in RunBuilder.get_runs(params):
	network = Network()
	loader = DataLoader(train_set batch_size=run.batch_size, num_workers=run.num_workers)
	optimizer = optim.Adam(networ.parameters(), lr=run.lr)
	
	m.begin_run(run, network, loader)
	for epoch in range(1):
		m.begin_epoch()
		for batch in loader:
			images, labels = batch
			preds = network(images)
			loss = F.cross_entropy(preds, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			m.track_loss(loss)
			m.track_num_correct(preds, labels)
		m.end_epoch()
	m.end_run()
m.save('results')
```
<a href="#top">return to top</a>
<a name="features"></a>
### Features

#### Batch Normalization

``` python
layers.add_module('conv_norm' + str(i), nn.BatchNorm2d(out_channels))
```

#### Regularization
``` python

```