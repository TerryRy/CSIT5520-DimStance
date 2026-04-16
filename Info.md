## run `python main.py` on root of this proect

## set `model_name`, `is_eval`, `pooling` and `batch_size` between line 21-26 of `main.py`

- `model_name`: model used in this project
- `is_eval`: If `True`, load weight from `f"./checkpoints/{model_name}_{pooling}.pth"`.
  specially, for `bert-base-multilingual-cased_False_32.pth` and `bert-base-multilingual-cased_False_64.pth`, set `batch_size` to the same number and delete "_32" or "_64" before running the code.
  And also, if `True`, it'll skip traing (obviously including saving weight and loss curve).
  If `False`, load pre-trained model and carry out the complete process.
- `pooling`: If `True`, activate aspect masking and mean pooling.
- `batch_size`: mostly 32 because 64 has high VRAM requirements. But `batch_size` must be 64 if loading `bert-base-multilingual-cased_False_64.pth`