__author__ = 'vincentpham'
import glob
from PIL import Image
import numpy as np
import csv
import time
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

files = glob.glob("coil-20-proc/*.png")
n = len(files)

#Inspecting png file
im = Image.open(files[0])
im.size #128x128

h2o.init()

train = h2o.import_file("obj_file_train.csv")
test = h2o.import_file("obj_file_test.csv")

y = train.names[1]
x = train.names[2:]

#Encode the response columns as categorical for multinomial classification
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

start_time = time.time()
# Train Deep Learning model  and validate on test set
model = H2OAutoEncoderEstimator(activation = "Tanh",
                                        hidden = [300,400,300],
                                        sparse = True,
                                        l1 = 1e-4,
                                        epochs = 100)


model.train(x=x,
            training_frame = train)

pred = model.predict(test)
pred.head() #not working?


test_np = test.as_data_frame()
test_n = len(test_np[0])
#test_np = test_np[1:]


pred_np = pred.as_data_frame()

for i in range(1,test_n):
    figure_array = [int(float(x[i])) for x in pred_np]
    W = np.reshape(figure_array,(128,128))
    data = np.array(W, dtype=np.uint8)
    img = Image.fromarray(data)
    img.save("autoencoded/" + test_np[0][i])


end_time = time.time()
total_time = round(end_time - start_time,2)
print(total_time)


h2o.shutdown()

#200x400x200            epoch = 10 -> 0.204861111111
#200x400x200            epoch = 100 -> [0.684027777778,0.739583333333] time = [,292.06]
#200x400x200x400        epoch = 100 -> [0.881944444444,0.715277777778] time = [,277.55]
#200,400,200,400,200    epoch = 100 -> 0.524305555556 time = 256.48

#300x500x300            epoch = 100 -> 0.9375           time = 409.3
#400x600x400            epoch = 100 -> 0.944444444444   time = 483.1

#400x600x400x600        epoch = 100 -> [0.989583333333,0.986111111111]   time = [552.86,460.22]
#400x600x400x600x600    epoch = 100 -> 0.913194444444   time = 488.3

#500x600x500x600        epoch = 100 -> 0.986111111111   time = 481.0