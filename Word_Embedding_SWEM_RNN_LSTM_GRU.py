## Part 1: Word Embeddings
## Part 2: Using word embedding in deep model (SWEM: Simple Word Embedding Model): model with pretraininded embeddings
## Part 3: Learning word Embedding using SWEM
## Part 4: RNN  LSTM and GRU (Gated recurrent unit)
# Source https://levelup.gitconnected.com/introduction-to-natural-language-processing-nlp-in-pytorch-8b7344c9dfec
# https://github.com/DenysLins/introduction_to_machine_learning/blob/master/4B_Natural_Language_Processing_Assignment.ipynb 4A Jupyter notebook 

"""
############################################################################################################
Part 1: Word Embedding  
#############################################################################################################

"""
# Download word vectors
from urllib.request import urlretrieve
import os
if not os.path.isfile('datasets/mini.h5'):
    print("Downloading Conceptnet Numberbatch word embeddings...")
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
    urlretrieve(conceptnet_url, 'datasets/mini.h5')

import h5py

with h5py.File('datasets/mini.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]
    
print("all_words dimensions: {}".format(len(all_words)))
print("all_embeddings dimensions: {}".format(all_embeddings.shape))

print("Random example word: {}".format(all_words[1337]))

english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
english_embeddings = all_embeddings[english_word_indices]

print("Number of English words in all_words: {0}".format(len(english_words)))
print("english_embeddings dimensions: {0}".format(english_embeddings.shape))

print(english_words[1337])

import numpy as np

norms = np.linalg.norm(english_embeddings, axis=1)
normalized_embeddings = english_embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1])

index = {word: i for i, word in enumerate(english_words)}
def similarity_score(w1, w2):
    score = np.dot(normalized_embeddings[index[w1], :], normalized_embeddings[index[w2], :])
    return score

# A word is as similar with itself as possible:
print('cat\tcat\t', similarity_score('cat', 'cat'))

# Closely related words still get high scores:
print('cat\tfeline\t', similarity_score('cat', 'feline'))
print('cat\tdog\t', similarity_score('cat', 'dog'))

# Unrelated words, not so much
print('cat\tmoo\t', similarity_score('cat', 'moo'))
print('cat\tfreeze\t', similarity_score('cat', 'freeze'))

# Antonyms are still considered related, sometimes more so than synonyms
print('antonym\topposite\t', similarity_score('antonym', 'opposite'))
print('antonym\tsynonym\t', similarity_score('antonym', 'synonym'))

def closest_to_vector(v, n):
    all_scores = np.dot(normalized_embeddings, v)
    best_words = list(map(lambda i: english_words[i], reversed(np.argsort(all_scores))))
    return best_words[:n]

def most_similar(w, n):
    return closest_to_vector(normalized_embeddings[index[w], :], n)

print(most_similar('cat', 10))
print(most_similar('dog', 10))
print(most_similar('duke', 10))

def solve_analogy(a1, b1, a2):
    b2 = normalized_embeddings[index[b1], :] - normalized_embeddings[index[a1], :] + normalized_embeddings[index[a2], :]
    return closest_to_vector(b2, 1)

print(solve_analogy("man", "brother", "woman"))
print(solve_analogy("man", "husband", "woman"))
print(solve_analogy("spain", "madrid", "france"))

"""
############################################################################################################
Part 2: Using weord embedding in deep model (SWEM: Simple Word Embedding Model)
#############################################################################################################

"""
import string
remove_punct=str.maketrans('','',string.punctuation)

# This function converts a line of our data file into
# a tuple (x, y), where x is 300-dimensional representation
# of the words in a review, and y is its label.
def convert_line_to_example(line):
    # Pull out the first character: that's our label (0 or 1)
    y = int(line[0])
    
    # Split the line into words using Python's split() function
    words = line[2:].translate(remove_punct).lower().split()
    
    # Look up the embeddings of each word, ignoring words not
    # in our pretrained vocabulary.
    embeddings = [normalized_embeddings[index[w]] for w in words
                  if w in index]
    
    # Take the mean of the embeddings
    x = np.mean(np.vstack(embeddings), axis=0)
    return x, y

# Apply the function to each line in the file.
xs = []
ys = []

with open("datasets/movie-simple.txt", "r", encoding='utf-8', errors='ignore') as f:
    for l in f.readlines():
        x, y = convert_line_to_example(l)
        xs.append(x)
        ys.append(y)

# Concatenate all examples into a numpy array
xs = np.vstack(xs)
ys = np.vstack(ys)

print("Shape of inputs: {}".format(xs.shape))
print("Shape of labels: {}".format(ys.shape))

num_examples = xs.shape[0]

print("First 20 labels before shuffling: {0}".format(ys[:20, 0]))

shuffle_idx = np.random.permutation(num_examples)
xs = xs[shuffle_idx, :]
ys = ys[shuffle_idx, :]

print("First 20 labels after shuffling: {0}".format(ys[:20, 0]))

import torch

num_train = 4*num_examples // 5

x_train = torch.tensor(xs[:num_train])
y_train = torch.tensor(ys[:num_train], dtype=torch.float32)

x_test = torch.tensor(xs[num_train:])
y_test = torch.tensor(ys[num_train:], dtype=torch.float32)

reviews_train = torch.utils.data.TensorDataset(x_train, y_train)
reviews_test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(reviews_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(reviews_test, batch_size=100, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F
class SWEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

## Training
# Instantiate model
model = SWEM()

# Binary cross-entropy (BCE) Loss and Adam Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(250):
    correct = 0
    num_examples = 0
    for inputs, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y = model(inputs)
        loss = criterion(y, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_examples += len(inputs)
    
    # Print training progress
    if epoch % 25 == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

## Testing
correct = 0
num_test = 0

with torch.no_grad():
    # Iterate through test set minibatchs 
    for inputs, labels in test_loader:
        # Forward pass
        y = model(inputs)
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))

# Check some words
words_to_test = ["exciting", "hated", "boring", "loved"]

for word in words_to_test:
    x = torch.tensor(normalized_embeddings[index[word]].reshape(1, 300))
    print("Sentiment of the word '{0}': {1}".format(word, torch.sigmoid(model(x))))


"""
############################################################################################################
Part 3: Learning the embeddings using SWEM  
#############################################################################################################

"""

VOCAB_SIZE = 5000
EMBED_DIM = 300

embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
embedding.weight.size()
class SWEMWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
model = SWEMWithEmbeddings(
    vocab_size = 5000,
    embedding_size = 300, 
    hidden_dim = 64, 
    num_outputs = 1,
)
print(model)



"""
############################################################################################################
Part 4: RNN  
#############################################################################################################

"""

mb = 1
x_dim = 300 
sentence = ["recurrent", "neural", "networks", "are", "great"]

xs = []
for word in sentence:
    xs.append(torch.tensor(normalized_embeddings[index[word]]).view(1, x_dim))
    
xs = torch.stack(xs, dim=0)
print("xs shape: {}".format(xs.shape))

h_dim = 128

# For projecting the input
Wx = torch.randn(x_dim, h_dim)/np.sqrt(x_dim)
Wx.requires_grad_()
bx = torch.zeros(h_dim, requires_grad=True)

# For projecting the previous state
Wh = torch.randn(h_dim, h_dim)/np.sqrt(h_dim)
Wh.requires_grad_()
bh = torch.zeros(h_dim, requires_grad=True)

print(Wx.shape, bx.shape, Wh.shape, bh.shape)

def RNN_step(x, h):
    h_next = torch.tanh((torch.matmul(x, Wx) + bx) + (torch.matmul(h, Wh) + bh))

    return h_next


# Word embedding for first word
x1 = xs[0, :, :]

# Initialize hidden state to 0
h0 = torch.zeros([mb, h_dim])

# Forward pass of one RNN step for time step t=1
h1 = RNN_step(x1, h0)

print("Hidden state h1 dimensions: {0}".format(h1.shape))
# Word embedding for second word
x2 = xs[1, :, :]

# Forward pass of one RNN step for time step t=2
h2 = RNN_step(x2, h1)

print("Hidden state h2 dimensions: {0}".format(h2.shape))
import torch.nn

rnn = nn.RNN(x_dim, h_dim)
print("RNN parameter shapes: {}".format([p.shape for p in rnn.parameters()]))
hs, h_T = rnn(xs)

print("Hidden states shape: {}".format(hs.shape))
print("Final hidden state shape: {}".format(h_T.shape))

lstm = nn.LSTM(x_dim, h_dim)
print("LSTM parameters: {}".format([p.shape for p in lstm.parameters()]))

gru = nn.GRU(x_dim, h_dim)
print("GRU parameters: {}".format([p.shape for p in gru.parameters()]))

