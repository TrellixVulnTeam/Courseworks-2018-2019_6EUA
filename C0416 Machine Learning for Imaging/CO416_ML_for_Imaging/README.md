## Machine Learning for Imaging - Lab Tutorials


### Python environment

We provide a pre-configured Python environment available from all lab machines which can be simply activated by calling

- on tcsh shell:
   ```shell
   source /vol/lab/course/416/venv/bin/activate.csh
   ```

- on bash shell:
   ```shell
   source /vol/lab/course/416/venv/bin/activate
   ```

If you want to setup a similar environment on your own computer, you can follow these steps (assuming you have conda installed).

#### 1. Setup and activate a Python 3.6 conda environment:

Make sure conda is up-to-date by running ```conda update conda```.

   ```shell
   conda create -n mli python=3.6
   source activate mli
   ```
   
Note: Try the *source* command in a *bash* shell, if it fails in another shell. 

#### 2. Install PyTorch:
   
   ```shell
   conda install pytorch torchvision -c pytorch
   ```

Details on pytorch installation can be found at [pytorch website](https://pytorch.org/get-started/locally/)

#### 3. Install other useful Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn scikit-learn SimpleITK
   ```
