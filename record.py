import nltk
#nltk.download('punkt')
import nltk.data
import pyaudio
import wave
import sys
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import time
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = input() 
print(url)
html = urlopen(url).read()
soup = BeautifulSoup(html)
soup.find('footer').decompose()
for div in soup.find_all("section", {'class':'bottom_detail'}): 
    div.decompose()# kill all script and style elements
for div in soup.find_all("section", {'class':'sidebar_2','class':'sidebar_3' }): 
    div.decompose()# kill all script and style elements

#soup.find('div', id="email-popup").decompose()

for script in soup(["script", "style", "span", "a", "label", "nav", "img", "em", "strong", "form", "button", "i"]):
    script.extract()    # rip it out

text = soup.body.get_text()
lines = (line.strip() for line in text.splitlines())
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
text = '\n'.join(chunk for chunk in chunks if chunk)
print(text)

dirname = os.path.dirname(__file__)
urltofilename = os.path.splitext(os.path.basename(url))[0]
filename = os.path.join(dirname, urltofilename+'.txt')
f = open(filename, "w+", encoding="utf8")
f.write(text)
f.close()



f = open(filename, "r", encoding="utf8")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
data = f.read()
listsentences = tokenizer.tokenize(data)

if not os.path.exists(urltofilename):
    os.makedirs(urltofilename)

filename = os.path.join(dirname, urltofilename)

descriptionfile = open(filename + "/description.txt" , "a", encoding="utf8")
descriptionfile.write(url+"\n")

index = 0

chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = chunk)

print("Press s to start record, q to quit record, esc to exit program")


all = []


def on_press(key):
    global index
    
    if key == keyboard.Key.esc:
        stream.close()
        p.terminate()
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys

    if k in ['s']:
        print('Key pressed: ' + k)
        print(listsentences[index])
        data = stream.read(chunk) # Record data
        all.append(data)

    if k in ['q']:
        print('Key pressed: ' + k)
        
        data = b''.join(all)
        filewavname = filename+"/"+str(index) + '.wav'
        wf = wave.open(filewavname, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
        descriptionfile.write(os.path.basename(filewavname)+"\n")
        descriptionfile.write(listsentences[index]+"\n")

        index+=1

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
listener.join()  # remove if main thread is polling self.keys

f.close()
descriptionfile.close()

