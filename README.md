# ECG Anomaly Detection
## Introductions
Greetings, fellow developers / researchers / enthusiasts! 

This repository aims to solve an Electrocardiogram (ECG) Arrhythmia (or eventually other heart problems) Detection by treating it entirely as an Anomaly Detection problem. 

Which means: we shall train our models with only the Normal class and use specific Anomaly Detection models, without using **any** data augmentation techniques whatsoever.

Isn't it exciting? :)

## Repository Tree
This is the current repository tree with all its necessary files. 

This structure shall not be changed unless there is **a new class implementation**. This is more clearly explained in the section [Guidelines for Developers](#guidelines-for-developers).

Any external files should always remain in the user's local machine, as Github is not intended to store multiple files. The only external files stored in this repository are meant to facilitate the user's intuition on its operation, which are located in the **dataset/signals** and **inputs/configs** directories.  

```bash
ECG-Anomaly-Detection/
├─ dataset/
│  ├─ signals/
│  │  ├─ 100.annot
│  │  ├─ 100.csv
│  ├─ mit_bih.py
├─ inputs/
│  ├─ configs/
│  │  ├─ default.conf
├─ source/
│  ├─ autoencoder.py
|  ├─ metrics.py
│  ├─ preprocess.py
│  ├─ test.py
│  ├─ train.py
├─ LICENSE
├─ README.md
├─ main.py
├─ requirements.txt
```


## Guidelines for Users
If you just mean to use this repository to generate your own results, here are a couple of simple instructions:


(**Note:** although not mandatory, it is recommended that you create a virtual environment to avoid possible conflicts with any packages that you may have in your system).

### Cloning the Repository 
```bash
$ git clone https://github.com/rbbh/ECG-Anomaly-Detection.git
```
### Installing necessary requirements
```bash
$ pip install -r requirements.txt
```
### Running the Code
This code runs using config files, as they are easy to edit via user level and easy to manipulate via developer level.

Here is an example of a config file, which corresponds to the default config file used in this project and is located in **inputs/configs/default.conf**:

```python
[SIGNALS]
mit_dir=100
channel=0

[PREPROCESS]
train_val_split_pct=0.98
pickle_name=mitdir_100_window_len_128_hann_channel_0
feature_type=spectrograms
signal_sample_amount=400
window_len=128
step_len=6
window=hann
wavelet=mexh

[ML-MODEL]
model=oc-svm

[DL-MODEL]
epochs=20
batch_size=64
learning_rate=1e-3
dense_neurons=32
checkpoint_pct=0.1
weights_path=
```

If you want to run the code experimenting with this config file, it is as simple as running
```
$ python main.py
```
, as it is already recognized as the default config file in the code level.


Running the code with a different config file is as simple as running the command:

```bash
$ python main.py -c inputs/configs/your_config_file.conf
```
You can always create your own config files. As you can see in the one displayed above, there are countless parameters to try on, so experiment away! :)

## Guidelines for Developers 
This section introduces some few rules for anyone who wants to contribute with the project and implement any features, documentations or bug fixes.

### Code Writing
This repository is made entirely with Python and so it follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines for Python development. Please give it a nice read before writing any code, unless you are an experienced Python developer. 

Also, the programming paradigm used in this project is *Object Oriented*, as it provides a good modularization and intuition, so please do not stray away from this paradigm.

### Code Documentation
For every new class or method implemented in this repository, one should **always** document it, otherwise the project would not be able to scale easily and help others understand your code, so please always document everything well.

The Docstring methodology adopted in this project is the [Numpy Docstring Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html), which is a great docstring convention for documenting methods and functions, as it is well organized, explainable and intuitive. The documentation for classes follow a very similar convention. 

Below is an example of this documentation extracted from the current code:
```python
    def __segment(self, signals, peaks, annotations):
        """Segments ECG signals into single beats.

        Parameters
        ----------
        signals : numpy-array
                  ECG time signals.
        peaks : numpy-array
                Indexes of the ECG peaks.
        annotations : numpy-array
                      Labels of each beat.

        Returns
        -------
        normal_beats : numpy-array
                       Segmented normal beats.
        abnormal_beats : numpy-array
                         Segmented anomaly beats.

        """
```

### Pushing New Features
Here I introduce a few methodologies regarding new releases. Anyone experienced with Software Engineering should feel some comfort with them.

When implementing new features or bug fixes, follow these steps:

1. Create a new branch **from the dev branch**;
2. When creating the new branch, name it like this: *task_description*. For example, if you were to implement a new feature which makes use of a Resnet model, it would go like this: *feature_resnet_implementation*;
3. After finishing the implementation and making sure it runs correctly, open a Pull Request to the dev branch, where I should review the changes. If everything is alright, the changes will be approved to the dev branch and, eventually, to the main branch.

### Pushing Files
As stated in a previous section, **any external files should always remain in the user's local machine**, as Github is not intended to store multiple files. The only external files stored in this repository are meant to facilitate the user's intuition on its operation, and they are not heavy. 

Therefore, any other external files attempted to be added to this git repository will de denied, unless agreed beforehand with the owner of the repository.

