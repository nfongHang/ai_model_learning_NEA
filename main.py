from model import *

model = Network([10, 128, 64, 10], None, "xa", None)


print(model.feed_forward([0,1,2,3,4,5,6,7,8,9]))