import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def calc_hist(img):
  hist = [0]  * 256

  for i in img:
    for j in i:
      hist[j] += 1

  return hist

def mean(li, hist):
  sum = n = 0
  for i in range(len(li)):
    sum += hist[li[i]] * li[i]
    n += hist[li[i]]
  
  u = sum / n
  return u

def variance(li,hist):
  u = mean(li, hist)
  s = s1 = 0
  for i in range(len(li)):
    s += (hist[li[i]] * ((li[i] - u)**2))
    s1 += hist[li[i]]
  var = (s / s1)

  return var

# miminising the intra - class variance

def otsu_intra_class(img):
  hist = calc_hist(img)
  t_optimal = -1
  minimum = None
  plt.bar(list(range(256)), hist)
  plt.show()
  for i in range(0,255):
      t  =  i
      w1 = 0
      w2 = 0
      for i in range(t + 1):
        w1 += hist[i]

      w2 = sum(hist) - w1

      w1 = w1 / sum(hist)

      w2 = w2 / sum(hist)

     
      L = list(range(0,t+1))

      R = list(range(t+1, 256))  

      icv = (w1 * variance(L,hist)) +  (w2 * variance(R,hist))

      if (minimum is None) or (icv < minimum):
        minimum = icv
        t_optimal = t


  new_img = [[0]*len(img[0]) for i in range(len(img))]

  for i in range(len(img)):
    for j in range(len(img[0])):
      if img[i][j] <= t_optimal :
        new_img[i][j] = 0

      else:
        new_img[i][j] = 255


  new_img = np.array(new_img, dtype = "uint8")
  plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
 
  plt.axis("off")
  plt.show()
  return new_img


def otsu_inter_class(img):
  hist = calc_hist(img)
  t_optimal = -1
  maximum = None
  plt.bar(list(range(256)), hist)
  plt.show()
  for i in range(0,255):
      t  =  i
      w1 = 0
      w2 = 0
      for i in range(t + 1):
        w1 += hist[i]

      w2 = sum(hist) - w1

      w1 = w1 / sum(hist)

      w2 = w2 / sum(hist)

      L = list(range(0,t+1))

      R = list(range(t+1, 256))  

      icv = w1 * w2 * ((mean(L,hist) - mean(R,hist))**2)

      if (maximum is None) or (icv > maximum):
        maximum = icv
        t_optimal = t


  new_img = [[0]*len(img[0]) for i in range(len(img))]

  for i in range(len(img)):
    for j in range(len(img[0])):
      if img[i][j] > t_optimal :
        new_img[i][j] = 255

      else:
        new_img[i][j] = 0


  new_img = np.array(new_img, dtype = "uint8")
  plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
 
  plt.axis("off")
  plt.show()
  return new_img







