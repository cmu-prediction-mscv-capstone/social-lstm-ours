import os
import numpy as np
from utils import DataLoader
import cv2
import matplotlib.pyplot as plt
gr_vsp_file=os.path.join(os.getcwd(),'data/crowds_zara01.vsp')
video_path=os.path.join(os.getcwd(),'data/crowds_zara01.avi')
#dataloader=DataLoader(gr_vsp_file=gr_vsp_file,video_path=video_path,batch_size=1,seq_length=3,pred_length=10)
all_color=np.random.rand(200,3)

def plot(x,y,y_pred):
    frames_path=os.path.join(os.path.dirname(video_path),os.path.basename(video_path).split('.')[0])
    frame=x[-1][0][3]
    img=cv2.imread(os.path.join(frames_path,"{}".format(frame)+".png"))
    plt.imshow(img)
    for index_frame in range(len(x)):
        for index_ped in range(len(x[index_frame])):
            #print(index_frame,index_ped)
            x_cord=x[index_frame][index_ped][0]
            y_cord=x[index_frame][index_ped][1]
            ped=x[index_frame][index_ped][2]
            x_cord=dataloader.width/2 *(1+ x_cord)
            y_cord=dataloader.height/2*(1 + y_cord)
            #print(x_cord,y_cord,ped)
            plt.plot(x_cord,y_cord,color=all_color[ped,:],markersize=5,marker='x')

    for index_frame in range(len(y)):
        for index_ped in range(len(y[index_frame])):
            x_cord=y[index_frame][index_ped][0]
            y_cord=y[index_frame][index_ped][1]
            ped=y[index_frame][index_ped][2]
            x_cord=dataloader.width/2 *(1+ x_cord)
            y_cord=dataloader.height/2*(1 + y_cord)
            plt.plot(x_cord,y_cord,color=all_color[ped,:],markersize=5,marker='.')

    for index_frame in range(len(y_pred)):
        for index_ped in range(len(y_pred[index_frame])):
            x_cord=y_pred[index_frame][index_ped][0]
            y_cord=y_pred[index_frame][index_ped][1]
            ped=y_pred[index_frame][index_ped][2]
            x_cord=dataloader.width/2 *(1+ x_cord)
            y_cord=dataloader.height/2*(1 + y_cord)
            plt.plot(x_cord,y_cord,color=all_color[ped,:],markersize=5,marker='^')
            #plt.plot(y[index_frame][index_ped][0],y[index_frame][index_ped][1],color=all_color[y[index_frame][index_ped][2],:],markersize=5,marker='x')
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

'''while(dataloader.epoch==0):
    x,y=dataloader.next_batch()
    for index in range(dataloader.batch_size):
        y_pred=np.zeros(6)
        plot(x[index],y[index],y_pred)'''
frames_path=os.path.join(os.path.dirname(video_path),os.path.basename(video_path).split('.')[0])
#print(frames_path)
if os.path.exists(frames_path):
    dataloader=DataLoader(gr_vsp_file=gr_vsp_file,video_path=video_path,batch_size=1,seq_length=3,pred_length=3)
else:
    dataloader=DataLoader(gr_vsp_file=gr_vsp_file,video_path=video_path,batch_size=1,seq_length=3,pred_length=3,forcePreProcess=True)
error=0
count=0
while(dataloader.epoch==0):
    x_batch,y_batch=dataloader.next_batch()
    for index in range(dataloader.batch_size):
        x=x_batch[index]
        y=y_batch[index]
        curr_frame_index=-1
        #all_ped=[]
        y_pred=[]
        for frame_index in range(1,dataloader.pred_length+1):
            y_pred_curr=[]
            for ped_index in range(len(x[curr_frame_index])):
                #print(ped_index)
                #print(x[curr_frame_index])
                #print(dataloader.curr_seq,len(x),len(x[0]),len(x[0][0]))
                ped=x[curr_frame_index][ped_index][2]
                for ped_index_prev in range(len(x[curr_frame_index-1])):
                    ped_prev=x[curr_frame_index-1][ped_index_prev][2]
                    if ped==ped_prev:
                        x_coord=x[curr_frame_index][ped_index][0] + (frame_index)*(x[curr_frame_index][ped_index][0]-x[curr_frame_index-1][ped_index_prev][0])
                        y_coord=x[curr_frame_index][ped_index][1] + (frame_index)*(x[curr_frame_index][ped_index][1]-x[curr_frame_index-1][ped_index_prev][1])
                        ped=x[curr_frame_index][ped_index][2]
                        y_pred_curr.append([x_coord,y_coord,ped])

                        for ped_index_gr in range(len(y[frame_index-1])):
                            ped_gr=y[frame_index-1][ped_index_gr][2]
                            if ped==ped_gr:
                                x_coord_gr=y[frame_index-1][ped_index_gr][0]
                                y_coord_gr=y[frame_index-1][ped_index_gr][1]
                                error+=np.linalg.norm([dataloader.height*(x_coord_gr-x_coord), dataloader.width*(y_coord_gr-y_coord)])
                                count+=1

                        break

            y_pred.append(y_pred_curr)
        plot(x,y,y_pred)

        #need x_prev of each pedestrian id

print("The average displacement error for seq_length {} and pred_length {} is {} for count {}".format(dataloader.seq_length,dataloader.pred_length,error/count,count))
#print(len(x[0]),len(y[0]))
