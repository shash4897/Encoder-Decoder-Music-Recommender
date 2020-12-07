from flask import Flask,jsonify,request,abort,make_response,render_template,redirect,url_for
import pandas as pd
import numpy as np
from keras.models import load_model
from scipy import spatial
from sys import setrecursionlimit
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__,static_folder='frontend\\static')

# Builds the KDTree
def build_sim_tree(song_vocab):
    sim_tree = spatial.KDTree(song_vocab_vecs)
    return sim_tree

#On start setup
max_seq_len = 245
max_playlist_len = 3
max_pred_iterations = 500
song_vocab = pd.read_csv("norm_songs.csv")
song_vocab_vecs = song_vocab.drop(['id'],axis=1).values
setrecursionlimit(10000)
sim_tree = build_sim_tree(song_vocab)
client_creds_mgr = SpotifyClientCredentials(client_id="7d27b18ba4f64f4bb3e1a9bcc223743e",client_secret="4eca52eefbc44e1c941a6664f5dda8bb")
spotify = spotipy.Spotify(client_credentials_manager=client_creds_mgr)

#Varun integration
songs = []
names = []
#RestFul service to help Choosing tracks to add to playlist
@app.route('/',methods = ['GET'])
def mainFunction():
    return render_template('index.html')
@app.route('/songQ',methods = ['POST'])
def add_list():

    name = request.form["track"]
    results = spotify.search(q=name, type='track')

    res = json.dumps(results)
    main = json.loads(res)
    if (len(main['tracks']['items'])>0):
        songid = main['tracks']['items'][0]['id']
        songName = main['tracks']['items'][0]['name']
        print(songName)
        songs.append(songid)
        names.append(songName)
    songid = -1
    return redirect(url_for('mainFunction'))


@app.route('/songQ',methods = ['GET'])
def disp_songId():
    return jsonify({
        "idlist" : songs,
        "names" : names
        })

@app.route('/search',methods=['POST'])
def search():
    if not request.json or not 'q' in request.json:
         abort(400)
    results = spotify.search(q=request.json['q'], type='track')
    results = json.dumps(results)
    results = json.loads(results)
    return jsonify(results),201

@app.route('/addSong',methods=['POST'])
def add_song():
    if not request.json or not 'id' in request.json:
         abort(400)
    if request.json['id'] not in songs:
        songs.append(request.json['id'])
    return jsonify({'msg':"Added song successfully"}),201
# RESTful service to get recommendations
@app.route('/recommends',methods=['GET'])
def recommendations():

    print(songs)
    idlist = songs

    #get audio features and convert to df
    playlist = song_vocab[song_vocab['id'].isin(idlist)]
    playlist = playlist.drop(['id'],axis=1)

    #perform preprocessing on df
    playlist_inp = playlist.values

    for i in range(max_seq_len-len(playlist_inp)):
        playlist_inp = np.vstack([playlist_inp,np.zeros(29)])
    playlist_inp = np.reshape(playlist_inp,(1,-1,29))

    #load inferencing model
    inf_enc,inf_dec = get_models("inf_encoder.h5","inf_decoder.h5")

    #get predictions
    pred_ids = decode_sequence(playlist_inp,inf_enc,inf_dec,idlist)
    #use spotify API and spotify ID to get JSON data
    pred_dict = spotify.tracks(pred_ids)

    while(len(songs)):
        songs.pop()

    return render_template('predictions.html',pred = pred_dict)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

#fetches inference models
def get_models(enc_path,dec_path):
    enc_model = load_model(enc_path)
    dec_model = load_model(dec_path)
    return enc_model,dec_model

# makes predictions using the inference model
def decode_sequence(input_seq,encoder_model,decoder_model,idlist):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    target_seq = prep_sos_seq(input_seq)
    print(target_seq)

    checklist = idlist

    stop_condition = False
    rec_song_ids = []
    c = 0

    while not stop_condition:

        reg, key, mode, timesig ,h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Process outputs to produce input for next time step
        songid,song_vec = getclosestsong(reg[0,-1,:],key[0,-1,:],mode[0,-1,:],timesig[0,-1,:])

        if songid not in checklist:
            rec_song_ids.append(songid)
            print("added a song : "+songid)
        checklist.append(songid)

        # Exit condition:  hit max sequence length
        if len(rec_song_ids) >= max_playlist_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = song_vec

        # Update states
        states_value = [h, c]

        c = c + 1

    return rec_song_ids

def prep_sos_seq(input_seq):

    inp_seq = input_seq[0]
    reg_idx = [0,1,2,3,4,17,18,21,22,28]
    key_idx = [5,6,7,8,9,10,11,12,13,14,15,16]
    mode_idx = [19,20]
    time_sig_idx = [23,24,25,26,27]

    inp_seq_reg = np.mean(np.take(inp_seq,reg_idx,axis=1),axis=0)

    inp_seq_key = np.zeros(12)
    inp_seq_key[np.argmax(np.sum(np.take(inp_seq,key_idx,axis=1),axis=0))] = 1

    inp_seq_mode = np.zeros(2)
    inp_seq_key[np.argmax(np.sum(np.take(inp_seq,mode_idx,axis=1),axis=0))] = 1

    inp_seq_timesig = np.zeros(5)
    inp_seq_timesig[np.argmax(np.sum(np.take(inp_seq,time_sig_idx,axis=1),axis=0))] = 1

    sos_vec = np.concatenate((inp_seq_reg[0:5],inp_seq_key))
    sos_vec = np.concatenate((sos_vec,inp_seq_reg[5:7]))
    sos_vec = np.concatenate((sos_vec,inp_seq_mode))
    sos_vec = np.concatenate((sos_vec,inp_seq_reg[7:9]))
    sos_vec = np.concatenate((sos_vec,inp_seq_timesig))
    sos_vec = np.append(sos_vec,inp_seq_reg[9])

    sos_vec = sos_vec.reshape((1,1,-1))

    return sos_vec

# returns the closest matching song for the given vector from the song vocabulary
def getclosestsong(reg,key,mode,timesig):
  feat_vec_pred = {
      'acousticness':reg[0],
      'danceability':reg[1],
      'duration_ms':reg[2],
      'energy':reg[3],
      'instrumentalness':reg[4],
      'key_0':key[0],
      'key_1':key[1],
      'key_2':key[2],
      'key_3':key[3],
      'key_4':key[4],
      'key_5':key[5],
      'key_6':key[6],
      'key_7':key[7],
      'key_8':key[8],
      'key_9':key[9],
      'key_10':key[10],
      'key_11':key[11],
      'liveness':reg[5],
      'loudness':reg[6],
      'mode_0':mode[0],
      'mode_1':mode[1],
      'speechiness':reg[7],
      'tempo':reg[8],
      'time_signature_0':timesig[0],
      'time_signature_1':timesig[1],
      'time_signature_3':timesig[2],
      'time_signature_4':timesig[3],
      'time_signature_5':timesig[4],
      'valence':reg[9]
  }

  feat_vec_pred = np.array(list(feat_vec_pred.values()))

  _ , idx = sim_tree.query(feat_vec_pred)

  songid =  song_vocab.iloc[idx]['id']

  songvec = song_vocab_vecs[idx].reshape((1,1,-1))

  return songid,songvec



if __name__ == '__main__':
    app.run(debug=True)
