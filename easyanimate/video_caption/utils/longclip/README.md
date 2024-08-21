# Long-CLIP
Codes in this directory are borrowed from https://github.com/beichenzbc/Long-CLIP/tree/4e6f5da/model.

We only modify the following code in [model_longclip.py](model_longclip.py) from
```python
@property
def dtype(self):
    return self.visual.conv1.weight.dtype
```
to
```python
@property
def dtype(self):
    # Fix: the VideoCLIP-XL inference.
    if hasattr(self, "visual"):
        return self.visual.conv1.weight.dtype
    else:
        return self.token_embedding.weight.dtype
```