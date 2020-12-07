# Music Recommendation System Using RNN based Encoder Decoder Model

## Abstract:

The goal of our project is to build a Content-Based Music Recommendation System using the power of modern deep learning techniques. It's purpose would be to suggest songs based on the user's current playlist. We will be using an Encoder Decoder Neural Network Architecture along with a KD Tree for similarity check to implement this. The project will also include a UI for the user to use the system.

Please note this is a WIP and may not be in a fully functioning state. This was created as part of a final semester project and may not be suitable for real world application just yet.

## Usage:

1. Run spotify_data.py to create the dataset from the spotify IDs and the spotify API. Make sure to enter your spotify API credentials in the file before you run it.
2. Run MRS_Colab to pre-process data, train the model and save the weights.
3. Copy the following files to 'frontend 2.0':
    - Dataset/norm_playlist.csv
    - Dataset/norm_songs.csv
    - Model/inf_encoder.h5
    - Model/inf_decoder.h5
4. Start the flask application by navigating to 'frontend 2.0' and running the following shell command:
```
$ export FLASK_APP=app.py ('C:\path\to\app>set FLASK_APP=app.py' for Windows CMD)
$ flask run
```
5. View on localhost

I hope to re-vamp the folder structure in the future and make it easier to use the project.
