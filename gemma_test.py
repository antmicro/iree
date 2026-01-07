#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["jax==0.8.0", "gemma"]
# ///

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from gemma import gm

import sys

print(sys.executable)
print(sys.version)

print(f"jax version={jax.__version__}")
jax.config.update("jax_platforms", "cpu_plugin")
# jax.config.update("jax_platforms", "cpu_client")
# jax.config.update("jax_platforms", "cpu")

def passing_jit():
    ## The below line run a passing JIT
    print(jax.jit(lambda x: x * 2)(3.14159))

def sample_gemma(model_path: Path):
    ## Uncomment the below section to run gemma
    model = gm.nn.Gemma3_270M()
    model_path = model_path.resolve()
    params = gm.ckpts.load_params(model_path / "gemma-3-270m")

    tokenizer = gm.text.Gemma3Tokenizer(model_path / "tokenizer.model")
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )

    print('Sampling')
    out0 = sampler.sample("roses are red")
    print('Done sampling')
    print(out0)

def jit_scatter():
    ## The below section exercise just the scatter operation
    buffer = jnp.zeros([1, 4096, 1, 256], dtype=jnp.bfloat16)
    start_index = 8
    index = jnp.zeros([1], dtype=jnp.int32) + start_index
    data = jnp.ones([1, 256, 1, 256], dtype=jnp.bfloat16)

    def scatter(operand, indicies, updates):
        dimension_numbers = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            # inserted_window_dims=(0, 2, 3),
            scatter_dims_to_operand_dims=(1, ),
        )
        return jax.lax.scatter(
            operand,
            indicies,
            updates,
            dimension_numbers,
            indices_are_sorted=True,
            unique_indices=True)

    print("scattering")
    print(buffer)
    print(index)
    print(data)
    out = jax.jit(scatter)(buffer, index, data)
    with jnp.printoptions(threshold=sys.maxsize):
        print(out[:, start_index - 2 : start_index + 258, :, :30])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=None, help="Path to the Gemma3 270M model")
    args = parser.parse_args()

    passing_jit()

    if args.model_path and args.model_path.exists():
        sample_gemma(args.model_path)
    else:
        jit_scatter()

if __name__ == "__main__":
    main()
