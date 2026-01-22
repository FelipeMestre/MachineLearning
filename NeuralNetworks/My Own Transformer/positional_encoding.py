import numpy as np

max_length = 512
embed_size = 64
sequence_lenght_array = np.arange(max_length)
embed_length_array = 2 * np.arange(embed_size // 2)
p, i = np.meshgrid(embed_length_array, sequence_lenght_array)
pos_emb = np.empty((1, max_length, embed_size))
pos_emb[0, :, ::2] = np.sin(p/(10000 ** (i / embed_size)))
pos_emb[0, :, 1::2] = np.cos(p/(10000 ** (i / embed_size)))

print(pos_emb.shape)
print(np.sin(p/(10000 ** (i / embed_size))).shape)