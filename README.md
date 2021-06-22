# Jaka to Melodia (What's the melody): Contrastive Predictive Coding applied to music audio data

In our team we had people wishing to explore self-supervised learning techniques, as well as people interested from long before on applying machine learning to music. Our project, ***Jaka to Melodia*** (*What's the melody*) was the solution to satisfy us all.

Aside of the cool-factor of working with music, we considered it an interesting use-case for deep self-supervised learning as in [Humphrey et al. (2012)](http://yann.lecun.org/exdb/publis/pdf/humphrey-jiis-13.pdf), Humphrey, Bello, and LeCun stated: 

<center><i>“Deep architectures often require a large amount of labeled data for supervised training, a luxury music informatics has never really enjoyed. Given the proven success of supervised methods, MIR </i>(music information retrieval) <i> would likely benefit a good deal from a concentrated effort in the curation of sharable data in a sustainable manner”. </i> </center>

But, rather than spending resources and long and tedious hours on handcrafting huge labeled data sets for music, why not leverage the power of self-supervised learning to put to good use the massive amounts of unlabeled music data we have accumulated through centuries of human artistic production?.

Our goal was then to use self-supervised algorithms to train a deep learning model to extract useful latent variables from unlabeled raw music audio data. The usefulness of such representation was assessed by using them later for some **downstream tasks** for which we had labels (E.g. classifying the instrument and/or note playing at a certain moment) hoping to get improved performance with less labeled data with respect to a fully supervised model.

As our self-supervised algorithm we chose ***Contrastive Predictive Coding (CPC)*** ([van den Oord et al., 2019](https://arxiv.org/pdf/1807.03748.pdf)), a self-supervised algorithm quite well-suited for sequential data as it is music, and which we found  interesting not just conceptually, but because it is related to ongoing research in our university.

## Data set

We used the MusicNet data set [(Thickstun et al., 2016)](https://arxiv.org/pdf/1611.09827.pdf), a collection of 330 freely-licensed classical music recordings, together with over 1 million annotated labels indicating the precise time of each note in every recording, the instrument that plays each note, and the note's position in the metrical structure of the composition, as well as some aditional meta data, such as the ensemble of instruments used for the composition, and the composer.

We chose MusicNet as its size was reasonable for our compute resources and purposes (~11 GB, 34 hours of music), it provided us with labels useful for multiple downstream tasks, and we thought that focusing on a single genre would make learning easier for the neural net as the data it would be trying to model was less diverse while still being reasonably complex (11 instruments arranged in 21 different ensembles).

As MusicNet had a sampling frequency of 44.1 kHz, which was hard to deal with with our compute resources, we downsampled it to 16 kHz. **Each observation ($x_t$) consisted then of 20480 samples** (1.28 seconds of audio). 



## Suggested Usage


1. Install requirements.

    ```shell
    pip install -r requirements.txt
    ```
    
2. There are several setups to run training that allow you to change the architecture of the neural nets. Use the `--help` flag. 

  For instance, for training a CPC model using a SincNet encoder, negative sampling from the sequence, and keeping at most 5 GB of data in memory use:

    ```shell
    python main.py --samplingType samesequence --useSincNet --maxChunksInMem 5
    ```
    
  For running on the downstream task of automatic music transcription using a previously trained CPC encoder stored in `logs/checkpoint3.pt` use:
    
    ```shell
    python main.py --supervised --load logs/checkpoint3.pt --maxChunksInMem 5
    ```
    
## Results

A lengthy description of our project and the obtained results can be found under `notebooks/JTM progress report.ipynb`.
