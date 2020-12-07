import os
import csv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Uses the Spotify API along with track ids in the CSV file to get the features

client_creds_mgr = SpotifyClientCredentials(client_id="<your client id>",client_secret="<your secret key>")
spotify = spotipy.Spotify(client_credentials_manager=client_creds_mgr)
basedir = "MRS/"
readdf = pd.read_csv(os.path.join(basedir,"Dataset/data_CSV/0-999.csv"))

track_features_list = []

def get_audio_features(rows):
    track_ids = []
    for index,row in rows.iterrows():
        #print(row)
        track = row['trackid']
        tokens = track.split(':')
        track_id = tokens[2]
        track_ids.append(track_id)
    features = spotify.audio_features(track_ids)
    for tf,(index,row) in zip(features,rows.iterrows()):
        tf.update({'artist_name':row['artist_name'],'track_name':row['track_name'],'pid':row['pid']})
    return features

for i in range(100,len(readdf.index),100):
    print(i)
    track_features_list.extend(get_audio_features(readdf.iloc[(i-100):i]))

writedf = pd.DataFrame(track_features_list)
writedf.to_csv(os.path.join(basedir,"Dataset/spotify_data.csv"),index = False)

df = pd.read_csv(os.path.join(basedir,"Dataset/spotify_data.csv"))
df = df.loc[:, df.columns != 'pid']
df = df.drop_duplicates()
df.to_csv(os.path.join(basedir,"Dataset/spotify_songs.csv"),index = False)
