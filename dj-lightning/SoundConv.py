#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = 'kira@-築城院 真鍳'

import pygame #-----------------------------------#
import numpy as np #------------------------------#
from os import remove #---------------------------#
from sys import argv #----------------------------#
from wave import open #---------------------------#
from pydub import AudioSegment #------------------#
from linear import load, dump, sv #---------------#
from struct import unpack #-----------------------#
from pyaudio import PyAudio #---------------------#
from collections import Counter #-----------------#


def play(file, test):
    """ SRC:: https://www.youtube.com/watch?v=AShHJdSIxkY
    """
    pygame.init()
    screenSize = (800, 600)
    screen = pygame.display.set_mode((screenSize),0)
    go = True

    #Define Colours
    color = {'WHITE': (255,255,255), 'GREEN': (0,255,0),'RED': (255,0,0),'BLUE': (0,0,255),
             'BLACK': (0,0,0), 'FUCHSIA': (255, 0, 255), 'GRAY': (128, 128, 128), 
             'LIME': (0, 128, 0), 'MAROON': (128, 0, 0), 'NAVYBLUE': (0, 0, 128),
             'OLIVE': (128, 128, 0), 'PURPLE': (128, 0, 128), 'TEAL': (0,128,128)}

    co = [color['WHITE'], color['GREEN'], color['RED'], color['BLUE'], color['BLACK'],
          color['FUCHSIA'], color['GRAY'], color[ 'LIME'], color['MAROON'], color['NAVYBLUE'],
          color['OLIVE'], color['PURPLE'], color['TEAL']]

    # TEXT
    x, y = 10, 10
    font = pygame.font.SysFont ("Arial", 40)
    text = font.render(file, True, (color['TEAL']))

    def upg(c):
        screen.fill(co[c])
        screen.blit(text, (x,y))
        pygame.display.update()

    to_fit = [[],[]]
    def apd(n, m, sli):
        sli[0].append(n)
        sli[1].append(m)

    # Sound--
    CHUNK = 1024
    wf = open(file, 'rb')
    p = PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                   channels=wf.getnchannels(),
                   rate = wf.getframerate(),
                   output=True)
    data = wf.readframes(CHUNK)
    c = 0

    # LOAD TRAINED MODEL
    loaded_model = load()
    # USE ONLY MODEL TO CHANGE COLOR IF WE HAVE TRAINED MODEL

    while data != '':
        try:
            stream.write(data)
            data = wf.readframes(CHUNK)
            data_int = unpack(str(4*CHUNK) + 'B', data) # Error while unpacking
            k = Counter(data_int).most_common(4)
            s = [i[0] for i in k]
            if c == 3: # get every 3th value to change color
                try:
                    # CHANGE COLOR WITH MODEL PREDICTION
                    pre = loaded_model.predict([s])[0]
                    print('predicted: -> {} -> {}'.format(pre, co[pre]))
                    apd(s, pre, to_fit)
                    upg(pre)
                except Exception as e:
                    #print('Err: ', e)
                    pass
                c = 0
            c += 1
        except Exception as e:
            #print('Err: ', e)
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

    # TRAIN MODEL
    X_ = np.array(to_fit[0])
    Y_ = np.array(to_fit[1])
    new_model = loaded_model.fit(np.array(X_), Y_)

    # SAVE MODEL
    dump(new_model)

    # SAVE DATA TO TEST MODELS
    if test:
        sv(X_, Y_)

def convert_to_wav(file, test):
    """ Convert music: mp3 -> wav
    """
    name, fmt = file.split('.')
    wname = name + '.wav'
    if fmt == 'mp3':
        conv = AudioSegment.from_mp3(file)
        conv.export(wname, format='wav')
        remove(file)
        play(wname, test)
    elif fmt == 'wav': play(file, test)
    else: print('format: mp3, wav') 

if __name__ == "__main__":
    test = False
    getfile = argv
    if 'test' in getfile: test = True
    if len(getfile) > 1: convert_to_wav(getfile[1], test)
    else: print('python3 dj.py music')
