import os
import torch
import torch.nn as nn
import transformers
from dotenv import load_dotenv

load_dotenv()

#
# Embeddings Base Class
#


class Embeddings(nn.Module):
	def __init__(self):
		super().__init__()
		self.emb_dim = None

	def __repr__(self):
		return f'<{self.__class__.__name__}: dim={self.emb_dim}>'


class TransformerEmbeddings(Embeddings):
	def __init__(self, lm_name, baseline):
		super().__init__()
		# load transformer
		self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True, add_prefix_space=True)
		self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
		self._is_baseline = baseline
		initial_embeddings = self._lm.get_input_embeddings()
		
		# move model to GPU if available
		if torch.cuda.is_available():
			self._lm.to(torch.device('cuda'))

		# add special tokens
		ner_labels = os.getenv(f"ENTITY_LABELS").split()
		domains = os.getenv(f"DOMAINS").split(' ')


		# if not baseline, add the domains as special tokens
		if not self._is_baseline:
			added_ner_labels = self._tok.add_special_tokens({'additional_special_tokens': self.get_special_tokens(ner_labels, domains)})
			#added_ner_labels = self._tok.add_special_tokens({'additional_special_tokens': self.get_special_tokens(ner_labels, None)}) #for new model
		else:
			added_ner_labels = self._tok.add_special_tokens({'additional_special_tokens': self.get_special_tokens(ner_labels)})

		special_tokens = self._tok.special_tokens_map
		

		self._lm.resize_token_embeddings(len(self._tok))

		# public variables
		self.emb_dim = self._lm.config.hidden_size

	def get_special_tokens(self, ner_labels, domains=None):

		special_tokens = []

		for label in ner_labels:
			special_tokens.append(f'<E1:{label}>')
			special_tokens.append(f'</E1:{label}>')
			special_tokens.append(f'<E2:{label}>')
			special_tokens.append(f'</E2:{label}>')

		if domains is not None:
			special_tokens = domains + special_tokens

		return special_tokens

	def embed(self, sentences):
		embeddings = []
		emb_words, att_words = self.forward(sentences)

		# gather non-padding embeddings per sentence into list
		for sidx in range(len(sentences)):
			embeddings.append(emb_words[sidx, :len(sentences[sidx]), :].cpu().numpy())
		return embeddings

	def forward(self, sentences):
		#print(self._tok.decode(self._tok.encode(sentences[0])))

		tok_sentences = self.tokenize(sentences)
		model_inputs = {k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in tok_sentences}

		# perform embedding forward pass
		model_outputs = self._lm(**model_inputs, output_hidden_states=True)

		# extract embeddings from relevant layer
		hidden_states = model_outputs.hidden_states  # tuple(num_layers * (batch_size, max_len, hidden_dim))
		emb_pieces = hidden_states[-1] # batch_size, max_len, hidden_dim

		# return model-specific tokenization
		return emb_pieces, tok_sentences['attention_mask'], tok_sentences.encodings

	def tokenize(self, sentences):
		# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
		tok_sentences = self._tok(
			[sentence.split(' ') for sentence in sentences],
			is_split_into_words=True, padding=True, truncation=True, return_tensors='pt'
		)
		# move input to GPU (if available)
		if torch.cuda.is_available():
			tok_sentences = tok_sentences.to(torch.device('cuda'))

		return tok_sentences


#
# Pooling Function
#


def get_marker_embeddings(token_embeddings, encodings, ent1, ent2):
	if torch.cuda.is_available():
		start_markers = torch.Tensor().to(torch.device('cuda'))
	else:
		start_markers = torch.Tensor()

	for embedding, word_id in zip(token_embeddings, encodings.word_ids):
		if word_id == ent1:
			start_markers = torch.cat([start_markers, embedding])
		elif word_id == ent2:
			start_markers = torch.cat([start_markers, embedding])
	return start_markers

