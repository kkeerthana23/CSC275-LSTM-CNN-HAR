import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
train_loss = [1.5061,1.0903,0.8321,0.6823,0.5897,0.4934,0.4184,0.4091,0.3629,0.3080, 0.2678, 0.2427, 0.2365, 0.2005, 0.1811, 0.1959, 0.1554, 0.1339,
         0.1150, 0.1114]
val_loss = [1.2345,0.9000,0.7184,0.6177,0.5250,0.4493,0.4087,0.4384,0.3166,0.3386,0.2391,0.1905,0.2605,0.1734,0.1669,0.1785,0.1608,0.1502,0.1074
        ,0.2236]
train_acc = [0.4341,0.6653,0.7438,0.7823,0.8110,0.8465,0.8673,0.8780,0.8909,0.9075,0.9229, 0.9283,0.9306,0.9452,0.9474,0.9447,0.9549,0.9603,
             0.9708,0.9743]
val_acc = [0.6039,0.7458,0.7753,0.8006,0.8399,0.8596,0.8806,0.8750,0.9087,0.8876,0.9228,0.9424,0.9242,0.9522,0.9579,0.9522,0.9537,0.9663,
           0.9803,0.9284]

# epochs = range(1,21)
# plt.plot(epochs,train_loss,label="TRAINING LOSS")
# plt.plot(epochs,val_loss,label="VALIDATION LOSS")
# plt.title("Training and Validation Loss")
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs,train_acc,label="TRAINING ACCURACY")
# plt.plot(epochs,val_acc,label="VALIDATION ACCURACY")
# plt.title("Training and Validation Accuracy")
# plt.legend()
# plt.show()
# =============================================================================
# matrix = [[ 98 ,  1,   0,   0,   1,   1,   0],
#  [  2,  54,   0,   1,   3   ,0,   0],
#  [  0,   0, 154,   0,   0   ,0 ,  0],
#  [  0,   0,   0,  45,   0  , 0  , 0],
#  [  0,   0,   0,   0,  76 ,  0  , 0],
#  [  0,   2,   1,   1,   0, 201,  0],
#  [  0,   0,   0,   0,   0,   1 , 70]]
# 
# fig, ax = plt.subplots(figsize=(8,8))
# sn.heatmap(matrix,annot=True, fmt='g', ax=ax)
# ax.set_xlabel("predicted labels");ax.set_ylabel("True labels")
# ax.set_title("confusion matrix")
# ax.xaxis.set_ticklabels(["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"])
# revlist = ["walk", "standup", "sitdown","run", "pickup","fall", "bed"]
# ax.yaxis.set_ticklabels(revlist)
# 
# 
# df = pd.DataFrame(matrix, index = [i for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]],
#                   columns = [i for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]])
# #plt.figure(figsize = (7,7))
# #sn.heatmap(matrix, annot=True)
# 
# =============================================================================





data = {'bed':1002,"fall":593, "pick":1535, "run":450,
        "sit":755, "stand":2042, "walk":705}
activities = list(data.keys())
label_count = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(activities, label_count, color ='maroon',
        width = 0.4)
 
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()
