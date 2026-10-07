"""lora_101: low-rank adaptation built from scratch on CPU.

The thesis: the update full fine-tuning applies to a weight matrix is approximately
low-rank, so freeze the base model and learn that update as the product of two thin
matrices (down-projection A, up-projection B), training 2*d*r params instead of d*d.
"""

__version__ = "0.1.0"
