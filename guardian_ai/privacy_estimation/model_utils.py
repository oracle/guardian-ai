import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
	def __init__(self, num_users, num_items, layers):
		super(MultiLayerPerceptron, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.factor_num = int(layers[0] / 2)
		self.layers = layers
		self.relu = nn.ReLU()
		self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
		self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)
		
		for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			var_name = "fc_layer_" + str(idx)
			setattr(self, var_name, nn.Linear(in_size, out_size))
		
		self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
		self.logistic = nn.Sigmoid()
	
	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
		for idx, _ in enumerate(range(len(self.fc_layers))):
			fc_layer = getattr(self, 'fc_layer_' + str(idx))
			vector = fc_layer(vector)
			vector = self.relu(vector)
		logits = self.affine_output(vector)
		rating = self.logistic(logits)
		return rating.squeeze()


class GeneralizedMatrixFactorization(nn.Module):
	def __init__(self, num_users, num_items, latent_dim):
		super(GeneralizedMatrixFactorization, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.latent_dim = latent_dim
		
		self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		
		self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
		self.logistic = torch.nn.Sigmoid()
	
	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		element_product = torch.mul(user_embedding, item_embedding)
		logits = self.affine_output(element_product)
		rating = self.logistic(logits)
		return rating.squeeze()


class NeuralCollaborativeFiltering(nn.Module):
	def __init__(self, num_users, num_items, layers, latent_dim):
		super(NeuralCollaborativeFiltering, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.latent_dim = latent_dim
		self.mlp_layers = int(layers[0] / 2)
		self.layers = layers
		self.relu = nn.ReLU()
		self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.mlp_layers)
		self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.mlp_layers)
		
		self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		
		for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			var_name = "fc_layer_" + str(idx)
			setattr(self, var_name, nn.Linear(in_size, out_size))
		self.affine_output = nn.Linear(in_features=self.layers[-1] + self.mlp_layers, out_features=1)
		self.logistic = nn.Sigmoid()
	
	def forward(self, user_indices, item_indices):
		user_embedding_mlp = self.embedding_user_mlp(user_indices)
		item_embedding_mlp = self.embedding_item_mlp(item_indices)
		
		user_embedding_mf = self.embedding_user_mf(user_indices)
		item_embedding_mf = self.embedding_item_mf(item_indices)
		
		mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
		mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
		
		for idx, _ in enumerate(range(len(self.fc_layers))):
			fc_layer = getattr(self, 'fc_layer_' + str(idx))
			vector = fc_layer(vector)
			vector = self.relu(vector)
		
		vector = torch.cat([mlp_vector, mf_vector], dim=-1)
		logits = self.affine_output(vector)
		rating = self.logistic(logits)
		return rating.squeeze()
