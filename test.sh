
### Dehaze
python test_SOTS_HINT.py
python evaluate_SOTS.py

### Derain
python test_rain100L.py

### Denoising
python test_gaussian_color_denoising_HINT.py --model_type blind
python evaluate_gaussian_color_denoising_HINT.py --model_type blind

### Desnowing
python test_snow100k.py
python evaluate_Snow100k.py

### Enhancement 
python test_from_dataset_LOLv2_Real.py
python test_from_dataset_LOLv2_Syn.py




