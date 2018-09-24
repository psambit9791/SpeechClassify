from scipy.io import wavfile
import pandas as pd
import os

ROOT_PATH = "../"
song_dict = {}
speech_type = {'rap':0, 'singing':1, 'speech':2}
index = 0
for root, dirs, files in os.walk(ROOT_PATH+"data/"):
    for file in files:
        if file.endswith('.wav'):
            song_id = os.path.join(root, file)
            fs, data = wavfile.read(song_id)
            song_dict[index] = (file.split('.')[0], data, fs, speech_type[root.split('/')[-2]])
            index += 1
song_df = pd.DataFrame.from_dict(song_dict, orient='index', columns=['Name', 'Data', 'Freq', 'Type'])

song_df.to_hdf(ROOT_PATH+'song_df.h5', key='song_df')
