# Image/ Audio Compression by PCA

|Name               |Email                        |
|-------------------|-----------------------------|
|Nguyễn Lê Hồng Hạnh|honghanh.nguyen2104@gmail.com|

### **How to run file**

**Note: You can see my code and results in this file zip, or also on [my github](https://github.com/HongHanh2104/maths4ai)**

1. PCA algorithm can be found in [pca.py](https://github.com/HongHanh2104/maths4ai/blob/master/HW3/pca.py).
2. Input image and audio can be found in [data folder](https://github.com/HongHanh2104/maths4ai/tree/master/HW3/data), with image filename is _4.jpg_, and audio filename is _godzilla_roar.wav_.
3. Compressed images and audios are shown in [images folder](https://github.com/HongHanh2104/maths4ai/tree/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/images) and [audios folder](https://github.com/HongHanh2104/maths4ai/tree/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/audios), respectively. In each folder, files are the results of compressing each above descripted image and audio  corresponding to k = 1, 6, 11, ... 
4. The details of compression rate is shown in [compress_rate_result folder](https://github.com/HongHanh2104/maths4ai/tree/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/compress_rate_result), with [image.csv](https://github.com/HongHanh2104/maths4ai/blob/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/compress_rate_result/image.csv) for image file and [audio.csv](https://github.com/HongHanh2104/maths4ai/blob/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/compress_rate_result/audio.csv) for audio file.
5. The process of compressing all these images and audios are shown in [PCA for data compression.ipynb](https://github.com/HongHanh2104/maths4ai/blob/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/HW3%20-%20PCA%20for%20data%20compression%20.ipynb). In this file, all the directory would be default, if you want to test with other files, please change the directory.

### **Analysis**
**Image Compression**

As the diagram shown in [PCA for data compression.ipynb](https://github.com/HongHanh2104/maths4ai/blob/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/HW3%20-%20PCA%20for%20data%20compression%20.ipynb), when k is lower than 40, the compression rate is acceptable. In my opinion, k = 11 with the compression rate of 0.284 is the limit for image sensing.

**Audio Compression**
As the diagram shown in [PCA for data compression.ipynb](https://github.com/HongHanh2104/maths4ai/blob/master/HW3%20-%20Image%20%26%20Audio%20Compression%20with%20PCA/HW3%20-%20PCA%20for%20data%20compression%20.ipynb), when k is lower than 150, the compression rate is acceptable. In my opinion, k = 51 with the compression rate of 0.324 is the limit for audio sensing.

