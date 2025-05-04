# Traditional-GAN-256x256
Using traditional GAN structure to generate 256x256 images

You can use the following commands with Miniconda3 to create and activate your GAN Python environment before start it: 

•	Press Ctrl + Alt + T to open the Linux Terminal

•	Deactivate the conda environment $ conda deactivate

•	Create the conda environment $ conda env create -f environment.yml 

•	Activate the conda environment $ conda activate GAN

We have prepared a file “train.py” that contains the training script 

•	Press Ctrl + Alt +T to open the Linux Terminal.

•	Execute the python file by $ python train.py

We saved well-trained models under “models/trained_model/*.pth”. Now let us generate a face image by using the trained model. The evaluation code is located in “eval.py”. To use the trained model, you can modify the path at the line in the file.
It will input the latent vector to generate a face image. Now:

•	Press Ctrl + Alt +T to open the Linux Terminal. 

•	Execute the python file by $ python eval.py
