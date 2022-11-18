# Dual Aggregation Transformer for Image Super-Resolution

This repository is for DAT introduced in the paper.

## Dependencies

- Python 3.8
- pytorch >= 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Cd to the default directory 'DAT'
pip install -r requirements.txt
python setup.py develop
```

## Test

- Download the pre-trained [models](https://ufile.io/4u0ms0h5) and place them in `experiments/pretrained_models/`.

  We provide all models: DAT, DAT-L, and DAT-2 (x2, x3, x4).

- Download [testing](https://ufile.io/6ek67nf8) (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the folloing scripts. The testing configuration is in `options/Test/`. More detail about YML, please refer to [Configuration](https://github.com/XPixelGroup/BasicSR/blob/master/docs/Config.md).

  **You can change the testing configuration in YML file, like 'test_DAT_x2.yml'.**

  ```shell
  # No self-ensemble
  # DAT, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/Test/test_DAT_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_x3.yml
  
  # DAT-L, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/Test/test_DAT_L_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_L_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_L_x3.yml
  
  # DAT-L, reproduces results in Table 1 of the supplementary material
  python basicsr/test.py -opt options/Test/test_DAT_2_x2.yml
  python basicsr/test.py -opt options/Test/test_DAT_2_x3.yml
  python basicsr/test.py -opt options/Test/test_DAT_2_x3.yml
  ```

- The output is in `results`.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).