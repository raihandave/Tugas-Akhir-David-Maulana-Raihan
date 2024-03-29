# -*- coding: utf-8 -*-
"""Kontras Hilal Tugas Akhir.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uz9hl65QSIviAeeLHBR4b_a5J_JqlWAs
"""

pip install skyfield

#library yang dipakai
import numpy as np
import math
import os
import datetime
import pandas as pd
import json
from skyfield.api import load
from skyfield.framelib import ecliptic_frame
from skyfield.api import load, wgs84, S, E

#panggil nama file
# path = '/content/drive/MyDrive/Sampel_data_hilal'
# name = '2023-08-11_23-14-18.png'
# hilalfile = os.path.join(path,name)
# time = datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S.png")

name = 'DATAWAKTU.txt'
data = open(name,'r')
dataload = data.readlines()
data.close()

#print(json.dumps(load, indent=2))

for line in dataload :
  newline = line.replace("\n","")
  #print(newline)
 # listline = newline.split(" ")
#  print(listline)
  time = datetime.datetime.strptime(newline,"   %d/%m/%Y %H:%M:%S")
print(time)

#dirfile = os.listdir(path)
#bikin list kosong
list_t = []
m_ra = []
m_dec = []
m_d = []
m_alt = []
salt =[]
m_az = []
m_w = []
m_arcv = []
m_daz = []
s_d = []
m_angle = []
arcl = []
m_age = []
m_illuminated = []

#Mulai Perhitungan
#for name in dirfile :

for line in dataload :
  newline = line.replace("\n","")
  time = datetime.datetime.strptime(newline,"   %d/%m/%Y %H:%M:%S")
 # time = datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S.png")
  ts = load.timescale()
  t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
  list_t.append(time)
  #t = ts.now()

  #panggil ephemeris dan objek yg dipake
  eph = load('de431t.bsp') #ini ephemerisnya
  sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
  e = earth.at(t)
  s = e.observe(sun).apparent()
  m = e.observe(moon).apparent()

  #lokasi Pengamatan
  top = wgs84.latlon(5.357194444 * S, 105.3115833 * E, elevation_m=90)

  #Alt Az Bulan dan matahari
  a = (earth + top).at(t).observe(moon).apparent()
  alt, az, distance, alt_rate, az_rate, range_rate = (a.frame_latlon_and_rates(top))

  b = (earth + top).at(t).observe(sun).apparent()
  s_alt, s_az, s_distance, s_alt_rate, s_az_rate, s_range_rate = (b.frame_latlon_and_rates(top))

  #Phase bulan
  _, slon, _ = s.frame_latlon(ecliptic_frame)
  _, mlon, _ = m.frame_latlon(ecliptic_frame)
  phase = (mlon.degrees - slon.degrees) % 360.0

  #umur Bulan
  age = (phase/360)*29.53

  #persen fase bulan
  percent = 100.0 * m.fraction_illuminated(sun)

  #RA dec dan elongasi
  position = earth.at(t).observe(moon)
  ra, dec, distance = position.radec()
  elongation = m.separation_from(s).degrees
  selongation = s.separation_from(m).degrees

  #Apparent diameter & Width
  Rmoon = 1737.4
  app_moon = np.degrees(np.arctan((Rmoon*2)/distance.km))
  w = app_moon * percent/100

  #ARCV DAZ
  arcv = alt.degrees-s_alt.degrees
  daz = az.degrees-s_az.degrees
  # .degrees buat ubah yg angle jd derajat, supaya bisa di operasiin

  #masukin ke list
  m_ra.append(ra._degrees)
  m_dec.append(dec.degrees)
  m_d.append(distance.au)
  salt.append(s_alt.degrees)
  m_alt.append(alt.degrees)
  m_az.append(az.degrees)
  m_w.append(w)
  m_arcv.append(arcv)
  m_daz.append(daz)
  s_d.append(s_distance.au)
  m_angle.append(phase)
  arcl.append(elongation)
  m_age.append(age)
  m_illuminated.append(percent)

print(M_ra)

#Bikin Jadi Tabel
dict = {'Waktu_(UTC)':list_t,
        'Sun_Alt':salt,
        'Moon_RA':m_ra,
        'Moon_Dec':m_dec,
        'Moon_Distance':m_d,
        'Moon_ALt':m_alt,
        'Moon_Az':m_az,
        'Moon_Width':m_w,
        'ARCV':m_arcv,
        'DAZ':m_daz,
        'Sun_Distance':s_d,
        'Moon_Angle':m_angle,
        'ARCL':arcl,
        'Moon_Age':m_age,
        'Illuminated':m_illuminated}
df = pd.DataFrame(dict)
df

df.head(10)

# #tabel diurut berdasar waktu
# df.sort_values(by=['Waktu (UTC)'])

df = pd.DataFrame(dict)

# saving the dataframe
df.to_csv('databulanMODEL.csv')