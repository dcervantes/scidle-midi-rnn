import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
from IPython import embed
from random import *
import random

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes


def txt2notes(txt):
    txt = txt.replace(' [', '\n \n[')
    txt = txt.replace('] ', ']\n \n')
    #txt = txt.replace(']\n', '\n')
    txt_list = txt.split('\n')
    txt_final = list()

    for i in range(len(txt_list)):
        #print txt_list[i]
        if txt_list[i] == ' ':
            txt_final.append(txt_list[i])
        elif " " in txt_list[i] and  "]" in txt_list[i] and "[" in txt_list[i]:
            txt_cord = txt_list[i].replace('[', '\n[')
            txt_cord = txt_cord.replace(']', ']\n')
            txt_cor_list = txt_cord.split('\n')
            for i in range(len(txt_cor_list)):
                if " " in txt_cor_list[i] and  "]" not in txt_cor_list[i] and "[" not in txt_cor_list[i]:
                    in_cord = txt_cor_list[i].split(' ')
                    for i in range(len(in_cord)):
                        txt_final.append(in_cord[i])
                else:
                    txt_final.append(txt_cor_list[i])
        else:
            txt_final.append(txt_list[i])
    return txt_final

model_file  =  open('rnn_midi_25_100_200000.dat', 'r')
model =  np.load(model_file)

chars= model['chars']
#hprev= model['hprev']
hprev = np.zeros((100,1))
Wxh=model['Wxh']
Whh=model['Whh']
bh=model['bh']
by=model['by']
Why=model['Why']
vocab_size=model['vocab_size'][0]

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
char2note = {
    '1':48,'!':49,'2':50,'@':51,'3':52,'4':53,'$':54,'5':55,'%':56,'6':57,
    '^':58,'7':59,'8':60,'*':61,'9':62,'(':63,'0':64,'q':65,'Q':66,'w':67,
    'W':68,'e':69,'E':70,'r':71,'t':72,'T':73,'y':74,'Y':75,'u':76,'i':77,
    'I':78,'o':79,'O':80,'p':81,'P':82,'a':83,'s':84,'S':85,'d':86,'D':87,
    'f':88,'g':89,'G':90,'h':91,'H':92,'j':93,'J':94,'k':95,'l':96,'L':97,
    'z':98,'Z':99,'x':100,'c':101,'C':102,'v':103,'V':104,'b':105,'B':106,
    'n':107,'m':108, '&':58, '#':51
}

init_char = random.choice(list(char2note))

sample_ix = sample(hprev, char_to_ix[' '], 400)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
txt = init_char + txt
txt = txt.replace('\n', ' ')
txt = txt.replace('  ', ' ')

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
program = randint(0, 79)
time = randint(40, 100)
track.append(Message('program_change', program=program)) # default piano



#txt = "wIPJJ [WrtIO] O [Yhl] v [iph] yQTE [WTT] [YTuPS] T [SPS] [O] [0EW] Q [WrqDG] r [^TP] [EIOaI] | P [Oa] E"
#txt_file  =  open('/home/dcervant/Desktop/music_sheets/index.html.1', 'r')
#txt = txt_file.read()
txt_list = txt.replace('\n', ' ')
txt_list = txt2notes(txt)
for i in range(len(txt_list)):
    if  len(txt_list[i]) > 1:
        cord = txt_list[i]
        if cord.startswith('[') and cord.endswith(']') and " " not in cord:
            cord = cord.replace('[', '')
            cord = cord.replace(']', '')
            for k in range(0, len(cord)):
                track.append(Message('note_on', note=char2note[cord[k]], velocity=62))
            for k in range(0, len(cord)):
                track.append(Message('note_off', note=char2note[cord[k]], velocity=0, time=time*2))
        elif cord.startswith('[') and cord.endswith(']') and " " in cord:
            cord = cord.replace('[', '')
            cord = cord.replace(']', '')
            cord = cord.replace(' ', '')
            for k in range(0, len(cord)):
                track.append(Message('note_on', note=char2note[cord[k]], velocity=100, time=time/4))
                track.append(Message('note_off', note=char2note[cord[k]], velocity=0, time=time))
        else:
            for k in range(0, len(cord)):
                if cord[k] in char2note:
                    track.append(Message('note_on', note=char2note[cord[k]], velocity=100, time=time*2))
                    track.append(Message('note_off', note=char2note[cord[k]], velocity=0, time=time*2)) 
                elif cord[k] == ' ':
                    track.append(Message('note_on', note=0, velocity=0, time=time))
                elif cord[k] == '|':
                    track.append(Message('note_on', note=0, velocity=0, time=time*2))
    else:
            if txt_list[i] == '|':
                track.append(Message('note_on', note=0, velocity=0, time=time*2))
            elif txt_list[i] == ' ':
                track.append(Message('note_on', note=0, velocity=0, time=time))
            elif txt_list[i] in char2note:
                track.append(Message('note_on', note=char2note[txt_list[i]], velocity=100, time=time))
                track.append(Message('note_off', note=char2note[txt_list[i]], velocity=0, time=time))
    #track.append(Message('note_on', note=0, velocity=0, time=time))

mid.save('song1.mid')
