import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from dataset import GestureMusicDataset
from model import GestureEncoder, MusicGenerator, Discriminator, VAE_TransGAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_len = 512
output_len = 938
batch_size = 10
epochs=5
lr=3e-4
alpha=0.1
gamma=8


gesture_encoder = GestureEncoder(
    num_layers=6, 
    n_heads=8, 
    in_dim=69, 
    hid_dim=256, 
    out_dim=1024, 
    max_len=input_len, 
    pos_ff_dim=512, 
    dropout=0.1
).to(device)
music_generator = MusicGenerator(in_dim=1024, output_len=output_len).to(device)
discriminator = Discriminator(
    num_layers=6, 
    hid_dim=128, 
    n_heads=4, 
    pos_ff_dim=512, 
    dropout=0.1, 
    output_len=output_len
).to(device)

dataset = GestureMusicDataset(gesture_dir='./video', audio_dir='./audio_mfcc', padding_len=512)
# Create a DataLoader to process dataset in batches
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

optim_E = Adam(gesture_encoder.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
optim_G = Adam(music_generator.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
optim_D = Adam(discriminator.parameters(), lr=lr*alpha, betas=(0.9, 0.98), eps=1e-9)

bce_loss_fn = nn.BCELoss().to(device)
mse_loss_fn = nn.MSELoss().to(device) 


def make_model(input_len, output_len, gesture_encoder, music_generator, discriminator):
    model = VAE_TransGAN(input_len, output_len, gesture_encoder, music_generator, discriminator)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
      
    return model


def compute_loss(model, gestures, musics):
    batch_size = gestures.size()[0]
    # define labels for MSE loss
    ones_label = torch.ones((batch_size, 1)).to(device)
    zeros_label_fake = torch.zeros((batch_size, 1)).to(device)
    zeros_label_noise = torch.zeros((batch_size, 1)).to(device)

    mean, log_var, real_score, fake_score, noise_score, real_layer, fake_layer = model.forward(gestures, musics)
    real_loss = bce_loss_fn(real_score, ones_label)
    fake_loss = bce_loss_fn(fake_score, zeros_label_fake)
    noise_loss = bce_loss_fn(noise_score, zeros_label_noise)
    
    # Loss: GAN, Reconstruction, KL Divergence
    loss_gan = real_loss + fake_loss + noise_loss
    loss_rec = mse_loss_fn(real_layer, fake_layer)
    loss_kl_div = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) / torch.numel(mean)

    return loss_gan, loss_rec, loss_kl_div


def train(pretrained=False):
    model = make_model(input_len, output_len, gesture_encoder, music_generator, discriminator)
    if pretrained:
        model.load_state_dict(torch.load('model_weights.pth'))
    for epoch in range(epochs):
        for i, (gestures, musics) in enumerate(data_loader):
            model.train()
            # Loss: Encoder, Generator, Discriminator
            gestures, musics = gestures.float().to(device), musics.float().to(device)
            # Backpropagation - Discriminator
            loss_gan, loss_rec, loss_kl_div = compute_loss(model, gestures, musics)
            loss_D = loss_gan
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            # Backpropagation - Encoder
            loss_gan, loss_rec, loss_kl_div = compute_loss(model, gestures, musics)
            loss_E = loss_kl_div + loss_rec
            optim_E.zero_grad()
            loss_E.backward()
            optim_E.step()
            # Backpropagation - Generator
            loss_gan, loss_rec, loss_kl_div = compute_loss(model, gestures, musics)
            loss_G = gamma * loss_rec - loss_gan
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
            
            if i % 5 == 0:
                print(f'Epoch: [{epoch+1}/{epochs}], Batch: [{(i+1)*batch_size}/{450}], Loss_E: {loss_E.item():.3f}, Loss_G: {loss_G.item():.3f}, Loss_D: {loss_D.item():.3f}')
        torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == '__main__':
    train()