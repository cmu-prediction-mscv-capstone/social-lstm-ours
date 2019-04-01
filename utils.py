import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
class DataLoader():
    def __init__(self,gr_vsp_file,video_path,batch_size,seq_length,pred_length,forcePreProcess=False,visualize_preprocessed=False):
        '''
        Reading from vsp file so that video and the current annotation can be coordinated
        '''
        self.gr_vsp_file=gr_vsp_file
        self.video_path=video_path
        #self.batch_size=batch_size
        #self.seq_length=seq_length
        self.forcePreProcess=forcePreProcess
        self.preprocessed_path=os.path.join(os.path.dirname(gr_vsp_file),os.path.basename(gr_vsp_file).split('.')[0])+"_interp.npz"
        self.frames_path=os.path.join(os.path.dirname(video_path),os.path.basename(video_path).split('.')[0])
        self.visualize_preprocessed=visualize_preprocessed
        self.frame_list_path=os.path.join(os.path.dirname(gr_vsp_file),os.path.basename(gr_vsp_file).split('.')[0])+"_framelist.pkl"
        if self.forcePreProcess:
            if not os.path.exists(self.frames_path):
                os.mkdir(self.frames_path)
                self.gen_video_frames(self.video_path,self.frames_path)
            else:
                self.height,self.width,_=cv2.imread(os.path.join(self.frames_path,"0.png")).shape
            self.preprocess(self.gr_vsp_file,self.height,self.width,self.preprocessed_path)
            self.frame_preprocess(self.preprocessed_path,self.frame_list_path)
            #self.frame_preprocess(self.preprocessed_path)
        else:
            self.height,self.width,_=cv2.imread(os.path.join(self.frames_path,"0.png")).shape
        if self.visualize_preprocessed:
            #self.visualize_preprocessed_fn(self.preprocessed_path,self.frames_path)

            self.visualize_preprocessed_list(self.frame_list_path,self.frames_path)


        self.load_preprocessed(self.frame_list_path,batch_size,seq_length,pred_length)


    def frame_preprocess(self,preprocessed_path,frame_list_path):
        '''Returns list of all information'''
        data=np.load(preprocessed_path)
        all_pedestrians_interp=data['all_pedestrians_interp']
        pedestrian_avail=data['pedestrian_avail']
        num_pedestrian=all_pedestrians_interp.shape[0]
        num_frames=all_pedestrians_interp.shape[1]
        all_frames=[]
        frame_list=[]
        num_peds=[]
        for frame in range(0,num_frames,10):
            curr_frame=[]
            frame_list.append(frame)
            for ped in range(num_pedestrian):
                if(pedestrian_avail[ped,frame]):
                    curr_frame.append([all_pedestrians_interp[ped,frame,0],all_pedestrians_interp[ped,frame,1],ped, frame])
            all_frames.append(curr_frame)
            num_peds.append(len(curr_frame))
        f=open(frame_list_path,"wb")
        pickle.dump((all_frames,frame_list,num_peds),f,protocol=2)
        f.close()
        #return all_frames


    def load_preprocessed(self,frame_list_path,batch_size,seq_length,pred_length):
        f = open(frame_list_path, 'rb')
        raw_data = pickle.load(f)
        f.close()
        self.data=raw_data[0]
        self.frame_list=raw_data[1]
        self.num_ped_list=raw_data[2]

        self.tot_seq=len(self.data)

        self.batch_size=batch_size
        self.seq_length=seq_length
        self.pred_length=pred_length
        self.num_batches=int((self.tot_seq-self.pred_length)/self.batch_size/self.seq_length)
        self.curr_batch=0
        self.curr_seq=0
        self.epoch=0

    def next_batch(self):
        batch_index=0
        x_batch=[]
        y_batch=[]
        while(batch_index<self.batch_size):
            x_batch.append(self.data[self.curr_seq:self.curr_seq+self.seq_length])
            y_batch.append(self.data[self.curr_seq+self.seq_length:self.curr_seq+self.seq_length+self.pred_length])
            self.curr_seq+=self.seq_length
            batch_index+=1
        self.curr_batch+=1
        if self.curr_batch==self.num_batches:
            self.curr_seq=0
            self.curr_batch=0
            self.epoch+=1
        return x_batch,y_batch

    def gen_video_frames(self,video_path,frames_path):
        vidcap=cv2.VideoCapture(video_path)
        success,image=vidcap.read()
        frame=0
        self.height,self.width,_=image.shape
        while success:
            if frame%100==0:
                print("Processing frame ",frame)
            cv2.imwrite(os.path.join(frames_path,"{}".format(frame)+".png"),image)
            success,image=vidcap.read()
            frame+=1

    def preprocess(self,gr_vsp_file,height,width,preprocessed_path):
        fp=open(gr_vsp_file,'r')
        lines=fp.readlines()
        fp.close()
        count=0
        no_splines=int(lines[count].split()[0])
        all_pedestrians=None
        count+=1
        for pedestrian in range(no_splines):
            print(lines[count])
            no_control_pts=int(lines[count].split()[0])
            count+=1
            frame_loaded=False
            x_all=[]
            y_all=[]
            for pts in range(no_control_pts):
                temp=lines[count].split()
                x,y,frame=float(temp[0]),float(temp[1]),int(temp[2])
                count+=1
                if all_pedestrians is None:
                    all_pedestrians=np.array([x,y,frame,pedestrian]).reshape((1,4))
                else:
                    all_pedestrians=np.vstack((all_pedestrians,[x,y,frame,int(pedestrian)]))
            #remove int and verify
            #also everything can be done here itself rather than two loops

        num_max_frame=int(all_pedestrians[:,2].max())+1
        num_pedestrian=int(all_pedestrians[:,3].max())+1
        all_pedestrians_interp=np.zeros((num_pedestrian,num_max_frame,2))
        pedestrian_avail=np.zeros((num_pedestrian,num_max_frame))
        for ped in range(num_pedestrian):
            curr_ped=all_pedestrians[all_pedestrians[:,3]==ped]
            for index in range(1,curr_ped.shape[0]):
                curr_frame=int(curr_ped[index,2])
                prev_frame=int(curr_ped[index-1,2])
                all_pedestrians_interp[ped,prev_frame:curr_frame+1,0]=2*np.linspace(curr_ped[index-1,0],curr_ped[index,0],int(curr_frame+1-prev_frame))/self.width
                all_pedestrians_interp[ped,prev_frame:curr_frame+1,1]=-2*np.linspace(curr_ped[index-1,1],curr_ped[index,1],int(curr_frame+1-prev_frame))/self.height
                pedestrian_avail[ped,prev_frame:curr_frame+1]=1
        np.savez(preprocessed_path,all_pedestrians_interp=all_pedestrians_interp,pedestrian_avail=pedestrian_avail,height=height,width=width)
    def visualize_preprocessed_fn(self,preprocessed_path,frames_path):
        data=np.load(preprocessed_path)
        all_pedestrians_interp=data['all_pedestrians_interp']
        pedestrian_avail=data['pedestrian_avail']
        height=data['height']
        width=data['width']
        num_pedestrian=all_pedestrians_interp.shape[0]
        num_frames=all_pedestrians_interp.shape[1]
        all_color=np.random.rand(num_pedestrian,3)
        for frame in range(num_frames):
            img=cv2.imread(os.path.join(frames_path,"{}".format(frame)+".png"))
            plt.imshow(img)
            for ped in range(num_pedestrian):
                if  pedestrian_avail[ped,frame]==1:
                    x_frame=width/2 *(1+ all_pedestrians_interp[ped,frame,0])
                    y_frame=height/2*(1 + all_pedestrians_interp[ped,frame,1])
                    plt.plot(x_frame,y_frame,color=all_color[ped,:],markersize=5,marker='o')
            plt.show(block=False)
            plt.pause(0.10)
            plt.clf()



    def visualize_preprocessed_list(self,frame_list_path,frames_path):
        f = open(frame_list_path, 'rb')
        data = pickle.load(f)
        f.close()
        print("here")
        #data=np.load(preprocessed_path)
        all_frames=data[0]
        frame_list=data[1]
        num_ped_list=data[2]
        height=self.height
        width=self.width
        all_color=np.random.rand(200,3)
        for frame_index,frame in enumerate(frame_list):
            img=cv2.imread(os.path.join(frames_path,"{}".format(frame)+".png"))
            plt.imshow(img)
            for index in range(len(all_frames[frame_index])):
                print(all_frames[frame_index][index][3],frame)
                x_frame=width/2 *(1+ all_frames[frame_index][index][0])
                y_frame=height/2*(1 + all_frames[frame_index][index][1])
                
                ped=all_frames[frame_index][index][2]
                plt.plot(x_frame,y_frame,color=all_color[ped,:],markersize=5,marker='o')
            plt.show(block=False)
            plt.pause(0.02)
            plt.clf()
