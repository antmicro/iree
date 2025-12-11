#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.8.0", "gemma"]
# ///

import pdb


import jax
import jax.numpy as jnp
from gemma import gm
import logging

import sys

print(sys.executable)
print(sys.version)

print(f"jax version={jax.__version__}")
jax.config.update("jax_platforms", "cpu_plugin")
#jax.config.update("jax_platforms", "cpu_client")

# print(jax.jit(lambda x: x * 2)(3.14159))

model = gm.nn.Gemma3_270M()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)

tokenizer = gm.text.Gemma3Tokenizer()
sampler = gm.text.Sampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)

print('Sampling')
out0 = sampler.sample("roses are red")
print('Done sampling')
print(out0)

# buffer = jnp.zeros([1, 4096, 1, 256], dtype=jnp.bfloat16)
# start_index = 8
# index = jnp.zeros([1], dtype=jnp.int32) + start_index
# data = jnp.ones([1, 256, 1, 256], dtype=jnp.bfloat16)

# def scatter(operand, indicies, updates):
#     dimension_numbers = jax.lax.ScatterDimensionNumbers(
#         update_window_dims=(0, 1, 2, 3),
#         inserted_window_dims=(),
#         # inserted_window_dims=(0, 2, 3),
#         scatter_dims_to_operand_dims=(1, ),
#     )
#     return jax.lax.scatter(
#         operand,
#         indicies,
#         updates,
#         dimension_numbers,
#         indices_are_sorted=True,
#         unique_indices=True)

# print("scattering")
# print(buffer)
# print(index)
# print(data)
# out = jax.jit(scatter)(buffer, index, data)
# with jnp.printoptions(threshold=sys.maxsize):
#     print(out[:, start_index - 2 : start_index + 258, :, :30])

