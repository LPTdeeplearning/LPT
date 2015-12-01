__author__ = 'vincentpham'
import glob
from PIL import Image
import numpy as np
import csv
import time
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

files = glob.glob("coil-20-proc/*.png")
n = len(files)
files_split__ = [x.split("__") for x in files]
obj_id = [[x[0].split("/")[1],x[1].split(".")[0]] for x in files_split__]


#Inspecting png file
im = Image.open(files[0])
im.size #128x128

#Split into 20% Testing and 80% training
for i in range(n):
    im = Image.open(files[i])
    pix = im.load()
    pix_array = np.array(im)
    pix_array = np.reshape(pix_array,-1)
    pix_list = pix_array.tolist()

    if i%5==0:
        with open("obj_file_test.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([obj_id[i][0]] + pix_list)
    else:
        with open("obj_file_train.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([obj_id[i][0]] + pix_list)
print "done writing"


h2o.init()

train = h2o.import_file("obj_file_train.csv")
test = h2o.import_file("obj_file_test.csv")

y = train.names[0]
x = train.names[1:]

#Encode the response columns as categorical for multinomial classification
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

start_time = time.time()
# Train Deep Learning model  and validate on test set
model = H2ODeepLearningEstimator(distribution="multinomial",
                                 activation = "RectifierWithDropout",
                                 hidden = [400,600,400,600],
                                 input_dropout_ratio= .2,
                                 sparse = True,
                                 l1 = 1e-5,
                                 max_w2 = 10,
                                 train_samples_per_iteration=-1,
                                 classification_stop=-1,
                                 stopping_rounds=0,
                                 epochs = 200)

model.train(x=x,
            y=y,
            training_frame = train,
            validation_frame = test)

pred = model.predict(test)
pred.head() #not working?


test_np = test.as_data_frame()[0]
test_np = test_np[1:]
pred_np = pred.as_data_frame()[0]
pred_np = pred_np[1:]

wrong = 0

for i in range(len(pred_np)):
    if test_np[i] != pred_np[i]:
        wrong += 1

acc = 1 - wrong/float(len(pred_np))
print(acc)

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