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

def spec_variance(li,hist):
  sum = n = 0
  for i in range(len(li)):
    sum += hist[li[i]] * li[i]
    n += hist[li[i]]
  
  u = sum / n
  s = 0
  for i in range(len(li)):
    s += (hist[li[i]] * ((li[i] - u)**2))

  var = (s / len(li))

  return var


def otsu_intra_class(img):
  hist = calc_hist(img)
  t_optimal = -1
  minimum = None
  plt.bar(list(range(256)), hist)
  plt.show()
  for i in range(1,255):
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

      icv = (w1 * spec_variance(L,hist)) +  (w2 * spec_variance(R,hist))

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
  print(t_optimal)
  return new_img

book1 = cv2.imread("bluelock.jfif", 0)
result1 = otsu_intra_class(book1)
cv2.imwrite("result.jpeg", result1)




