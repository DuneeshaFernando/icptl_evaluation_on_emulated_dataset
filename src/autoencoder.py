import torch
import torch.nn as nn
import src.trainer_utils as utils
from collections import OrderedDict

device = utils.get_default_device()

class AutoEncoder(nn.Module):
    def __init__(self, in_size, latent_size, num_layers):
        super().__init__()

        num_neurons=[]
        for l in range(num_layers):
            num_neurons.append(in_size)
            in_size=int(in_size/2)
        num_neurons.append(latent_size)

        encoder_layers = OrderedDict()
        for layer_n in range(num_layers):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n+1])
            encoder_layers['layer_'+str(layer_n)] = h_layer
            encoder_layers['relu_'+str(layer_n)] = nn.ReLU()

        decoder_layers = OrderedDict()
        for layer_n in range(num_layers,0,-1):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n-1])
            decoder_layers['layer_' + str(layer_n)] = h_layer
            if layer_n == 1:
                decoder_layers['sigmoid'] = nn.Sigmoid()
            else:
                decoder_layers['relu_' + str(layer_n)] = nn.ReLU()

        self.encoder = nn.Sequential(encoder_layers).to(device)
        self.decoder = nn.Sequential(decoder_layers).to(device)

    def forward(self, input_window):
        latent_window = self.encoder(input_window)
        reconstructed_window = self.decoder(latent_window)
        return reconstructed_window

def training(epochs, autoencoder_model, train_loader, val_loader, test_loader, learning_rate, model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    test_loss_dict = {}

    # Evaluate before starting training - Epoch 0
    with torch.no_grad():
        test_loss = 0
        for [test_batch] in test_loader:
            test_batch = utils.to_device(test_batch, device)
            test_recon = autoencoder_model(test_batch)
            t_loss = criterion(test_recon, test_batch)
            test_loss += t_loss.item() * test_batch.shape[0]
        test_loss = test_loss / len(test_loader.dataset)
        print(f'Epoch:{0}, Test Loss: {test_loss:.4f}')
        test_loss_dict[0] = test_loss

    val_loss_list = []

    for epoch in range(epochs):
        train_loss = 0
        for [batch] in train_loader:
            batch = utils.to_device(batch, device)
            recon = autoencoder_model(batch)
            loss = criterion(recon, batch)
            train_loss += loss.item() * batch.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)
        # print(f'Epoch:{epoch + 1}, Loss: {loss.item():.4f}')

        with torch.no_grad():
            val_loss = 0
            for [val_batch] in val_loader:
                val_batch = utils.to_device(val_batch, device)
                val_recon = autoencoder_model(val_batch)
                v_loss = criterion(val_recon, val_batch)
                val_loss += v_loss.item() * val_batch.shape[0]
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)

        print(f'Epoch:{epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving best model ..")
            # Save the model
            torch.save({
                'encoder': autoencoder_model.encoder.state_dict(),
                'decoder': autoencoder_model.decoder.state_dict()
            }, model_name)

        # Also evaluate the test loader
        with torch.no_grad():
            test_loss = 0
            for [test_batch] in test_loader:
                test_batch = utils.to_device(test_batch, device)
                test_recon = autoencoder_model(test_batch)
                t_loss = criterion(test_recon, test_batch)
                test_loss += t_loss.item() * test_batch.shape[0]
            test_loss = test_loss / len(test_loader.dataset)
            print(f'Epoch:{epoch + 1}, Test Loss: {test_loss:.4f}')
            test_loss_dict[epoch + 1] = test_loss

        # Algorithmic stop : If val_loss is not changing significantly for a slack of 5, break from the for loop
        if len(val_loss_list) >= 5 and round(val_loss_list[-1], 4) == round(val_loss_list[-5], 4):
            return test_loss_dict
            break

    return test_loss_dict

def testing(autoencoder_model, test_loader):
    results = []
    results_rca = []
    for [batch] in test_loader:
        batch = utils.to_device(batch, device)
        with torch.no_grad():
            recon = autoencoder_model(batch)
        results.append(torch.mean((batch - recon) ** 2, axis=1))
    return results