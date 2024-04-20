import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
	def __init__(self, num_users, num_items, layers):
		super(MultiLayerPerceptron, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_num = int(layers[0] / 2)
		self.layers = layers
		self.relu = nn.ReLU()
		self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_num)
		self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_num)
		self.logistic = nn.Sigmoid()
		
		for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			layer_name = "mlp_layer_" + str(idx)
			setattr(self, layer_name, nn.Linear(in_size, out_size))
		
		self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
	
	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		vector = torch.cat([user_embedding, item_embedding], dim=-1)
		for idx, _ in enumerate(range(len(self.layers) - 1)):
			mlp_layer_layer = getattr(self, 'mlp_layer_' + str(idx))
			vector = mlp_layer_layer(vector)
			vector = self.relu(vector)
		logits = self.affine_output(vector)
		rating = self.logistic(logits)
		return rating.squeeze()


class GeneralizedMatrixFactorization(nn.Module):
	def __init__(self, num_users, num_items, embedding_dim):
		super(GeneralizedMatrixFactorization, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		
		self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
		self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
		
		self.affine_output = torch.nn.Linear(in_features=self.embedding_dim, out_features=1)
		self.logistic = torch.nn.Sigmoid()
	
	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		element_product = torch.mul(user_embedding, item_embedding)
		logits = self.affine_output(element_product)
		rating = self.logistic(logits)
		return rating.squeeze()


class NeuralCollaborativeFiltering(nn.Module):
	def __init__(self, num_users, num_items, layers, embedding_dim):
		super(NeuralCollaborativeFiltering, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_dim = embedding_dim
		self.mlp_layers = int(layers[0] / 2)
		self.layers = layers
		self.relu = nn.ReLU()
		
		self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.mlp_layers)
		self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.mlp_layers)
		
		self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
		self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
		
		self.mlp_layer_layers = nn.ModuleList()
		for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
			self.mlp_layer_layers.append(nn.Linear(in_size, out_size))
		
		self.affine_output = nn.Linear(in_features=self.layers[-1] + self.embedding_dim, out_features=1)
		self.logistic = nn.Sigmoid()
		
		self.init_weights()
	
	def init_weights(self):
		nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
		nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
		nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
		nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
		
		for mlp_layer_layer in self.mlp_layer_layers:
			nn.init.xavier_uniform_(mlp_layer_layer.weight)
		
		nn.init.xavier_uniform_(self.affine_output.weight)
		nn.init.zeros_(self.affine_output.bias)
	
	def forward(self, user_indices, item_indices):
		user_embedding_mlp = self.embedding_user_mlp(user_indices)
		item_embedding_mlp = self.embedding_item_mlp(item_indices)
		
		user_embedding_mf = self.embedding_user_mf(user_indices)
		item_embedding_mf = self.embedding_item_mf(item_indices)
		
		mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
		for mlp_layer_layer in self.mlp_layer_layers:
			mlp_vector = self.relu(mlp_layer_layer(mlp_vector))
		
		mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
		vector = torch.cat([mlp_vector, mf_vector], dim=-1)
		
		logits = self.affine_output(vector)
		rating = self.logistic(logits)
		return rating.squeeze()
