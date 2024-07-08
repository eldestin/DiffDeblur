# Residual Diffusion Deblurring Model for Single Image Defocus Deblurring
This is the repository of our project: Residual Diffusion Deblurring Model for Single Image Defocus Deblurring.

## Pipeline
![](./Figures/Model%20Structure.png)

## Dependencies
```
pip install -r requirements.txt
```

## Download the training and evaluation datasets
[DPDD dataset](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

[RealDof dataset](https://github.com/codeslake/IFAN)

[CUHK dataset](https://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html)

## Training
```
python main.py
```

## Evaluation
Choose the testing dataset in Evaluation folder and run:
```
cd Evaluation
python evaluation.py 
```

## Visual comparison:
![](./Figures/Comparison(DPDD).png)