#!/anaconda3/bin/python3.7

"""
End - End pipeline for assessing
Wireless Capsule Endoscopy Video
"""

import concurrent.futures
import os
import time
import subprocess
from datetime import datetime
import shutil
from parallel import mp_pool
from v2f import GetFrames, GetVideo
from demo_test import _test
from flask_cors import CORS

import boto3
from flask import Flask, render_template, request, send_from_directory, jsonify


class classification:

    """
    creating folders for processing
    the video and the images in steps
    of three.

    Video: contains the uploaded video
    Images: contains the frames of the
            uploaded video.

    step1: classification of images into
            Normal vs Abnormal
            (in respective folders).

    step2: reads all the Abnormal images,
            classifies into 9 classes
            and stores in respective
            Abnormality folders.

    step3: reads the images of different
            classes, segments and stores
            the predicted map in respective
            folders.

    """

    def __init__(self, patient_id, video_path):
        super(classification, self).__init__()

        """
        creates respective folders inside
        a master folder named with current
        time and date.
        """
        self.cwd = os.getcwd()
        self.patient_id = patient_id
        self.video_path = video_path
        # self.date = datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
        self.root = self.cwd + "/patients/" + self.patient_id
        self.video = "./patients/" + self.video_path

    def timestamp(self, patient_id, videoname):
        self.old_video = f'{self.cwd}/patients/{patient_id}/Videos/{videoname}'
        self.new_video = videoname.split('.')[0]+ '_time.' + videoname.split('.')[1]
        self.new_video_filepath = f'{self.cwd}/patients/{patient_id}/Videos/{self.new_video}'
        self.command = f"ffmpeg -i {self.old_video} -vf \"drawtext=x=8:y=8:box=1:fontfile='./Helvetica.ttf':fontcolor=white:boxcolor=black: \\expansion=strftime:basetime=$(%s -d'2013-12-01 12:00:00')000000: \\timecode='00\:00\:00\:00':rate=1:fontcolor='white'\" {self.new_video_filepath}"
        os.system(self.command)
        return self.new_video

    def create_folders(self):

        os.makedirs(os.path.join(self.root, "Videos"))
        os.makedirs(os.path.join(self.root, "Images"))
        os.makedirs(os.path.join(self.root, "step_3/Segmentation"))
        ab_list = ["Apthae", "Ulcer", "Bleeding", "Lymphangectasias", "Angioectasias",
                    "Polypoids", "ChylousCysts", "Stenoses", "Voedemas"]
        for i in range(1, 4):

            if i == 1:
                os.makedirs(os.path.join(self.root, "step_" + str(i)))
                os.makedirs(os.path.join(self.root + "/step_1", "Normal"))
                os.makedirs(os.path.join(self.root + "/step_1", "Abnormal"))

            if i == 2:
                os.makedirs(os.path.join(self.root, "step_" + str(i)))
                for j in range(9):
                    os.makedirs(os.path.join(self.root + "/step_2",
                    ab_list[j]))
            
        
        # shutil.copyfile("WCE_video.avi", self.root + "/Videos/WCE_video.avi")

    def _convert_video2img(self, filename):

        """
        function converting videos to frames
        """
        video = GetFrames(self.root + "/Videos/" + filename, self.root + "/Images")
        video.get_frame_names()
        frames = video.frame_names()


        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(video.save_frames, frames)
        video.subfolders()
        os.chdir(self.cwd)
        print("Video 2 Image conversion --> DONE")
    
        """
        classifying input images as Normal & Abnormal
        and stores in respective folders inside step_1
        """
        mp_pool()
        print("Classification of WCE to normal vs abnormal --> DONE")

    def _abnormality(self):

        """
        reads images from Abnormality folder in step_1,
        classifies and stores in step_2 respective
        Abnormality_{pred_class} folders
        """
        weights = [f"{self.cwd}/step2/inceptionv3_level1.pth",
                   f"{self.cwd}/step2/inceptionv3_apthae_ulcer_bleed_angio_lymph.pth",
                   f"{self.cwd}/step2/inceptionv3_poly_cyst_stenoses_voedemas.pth",
                   f"{self.cwd}/step2/inceptionv3_apthae_ulcer.pth",
                   f"{self.cwd}/step2/inceptionv3_bleed_angio_lymph.pth",
                   f"{self.cwd}/step2/inceptionv3_poly_cyst_stenoses.pth",
                   f"{self.cwd}/step2/inceptionv3_angio_lymph.pth",
                   f"{self.cwd}/step2/inceptionv3_poly_cyst_test.pth"
        ]
        _t = _test(self.root+"/step_1/Abnormal", self.root+"/step_2", weights, step=2)
        _t._predict()

        print("Classification of abnormal to 9 different classes --> DONE")

    def _segmentation(self):

        """
        reads images from step_2 of Abnormalities_{pred_class}
        iteratively and stores the segmented results in
        respective Abnormality folders in step_3
        """
        ab_list = ["Apthae", "Ulcer", "Bleeding", "Lymphangectasias", "Angioectasias",
                    "Polypoids", "ChylousCysts", "Stenoses", "Voedemas"]

        for i in range(8):
            os.makedirs(self.root + "/step_3/Segmentation/" + ab_list[i])
            if len(os.listdir(self.root + '/step_2/'+ ab_list[i])) != 0:
                _t = _test(
                    self.root + "/step_2/"+ ab_list[i],
                    self.root + "/step_3/Segmentation/" + ab_list[i],
                    f"{self.cwd}/step3/Abnormality_{i}/best_weights.pth.tar",
                    step=3,
                )
                # print('Abnormality: ', i)
                _t._predict()
            
            else:
                continue 

        print("Segmentation of abnormalities --> DONE")

    def _convert_img2video(self):

        ftv = GetVideo(
            path_to_frames = self.root + "/Images/dir_1/",
            save_video_to = self.root + "/Videos/",
            desired_fps = 1,
        )
        ftv.get_video()

        print("Image 2 Video conversion --> DONE")

if __name__ == "__main__":

    app = Flask(__name__)
    CORS(app)
    
    @app.route("/")
    def main_page():
        return render_template("m.html")

    @app.route("/", methods=["POST"])
    def main_page_request():
        t1 = time.time()
        patient_id = request.form["patientId"]
        filename = request.form["filename"]

        Bucket_name="spectrumlabiisc"
        #Upload a file for demo
        s3 = boto3.resource('s3')
        # filename = 'WCE_video.avi'
        filepath = patient_id+'/'+filename 
        # s3.meta.client.upload_file(video_path, Bucket_name, filepath)

        #Create an object of Classification - thereby the patient folder structure
        prepend = os.getcwd()
        local_file_path= prepend+'/patients/'+ patient_id+'/Videos/'+ filename
        endoscopy = classification(patient_id, local_file_path)   
        endoscopy.create_folders()
        #Download file from S3 bucket to local system p101/Videos/
        bucket = s3.Bucket(Bucket_name)
        object = bucket.Object(filepath)  # image name
        object.download_file(local_file_path)
        
        filename= endoscopy.timestamp(patient_id, filename) #Newly added along with the function 

        endoscopy._convert_video2img(filename)
        t2 = time.time()
        print("CONVERTING V2F TIME: ", t2-t1)
        #endoscopy._normal_abnormal()
        t3 = time.time()
        print("FIRST STEP TIME: ", t3-t2)
        endoscopy._abnormality()
        t4 = time.time()   
        print("SECOND STEP TIME: ", t4-t3)
        endoscopy._segmentation()
        t5 = time.time()
        print("THIRD STEP TIME", t5-t4)

        # endoscopy._convert_img2video()
        
        #Upload Keyframe Video to s3 bucket p101/
        # try:
        #     s3.meta.client.upload_file(f'{prepend}/patients/{patient_id}/Videos/key_frame.avi', Bucket_name, f'{patient_id}/key_frame.avi')
        #     # s3.meta.client.upload_file('step3_pred.avi', Bucket_name, patient_id+'/key_frame.avi')
        #     print("Upload Successful")
        # except FileNotFoundError:
        #     print("The file was not found")
            
        # print(patient_id, video_path)
        
       #Upload Keyframe Images to s3 bucket  
        result = []
        try:
            # root_path = f'{prepend}/patients/{patient_id}/step_3/' # local folder for upload
            seg_path = f'{prepend}/patients/{patient_id}/step_3/Segmentation/'
            org_path = f'{prepend}/patients/{patient_id}/step_2/'
            path = f'{prepend}/patients/{patient_id}/'
            my_bucket = s3.Bucket(Bucket_name)
            
            #Uploading segmented images
            for path, subdirs, files in os.walk(seg_path):
                #directory_name = f'{patient_id}/Output Images'
                for file in files:
                    abnormality_name = path.split(f'/{patient_id}/step_3/Segmentation/')[1] 
                    directory_name = f'WCE/{patient_id}/Output Images/' + f'{abnormality_name}' 
                    my_bucket.upload_file(os.path.join(path, file), directory_name+'/'+file)
                    result.append({
                        "abnormality" : abnormality_name,
                        "key" : directory_name+'/'+file
                    })
            
            #Uploading original images 
            for path, subdirs, files in os.walk(org_path):                
                for file in files:
                    abnormality_name = path.split(f'/{patient_id}/step_2/')[1] 
                    directory_name = f'WCE/{patient_id}/Original Images/' + f'{abnormality_name}' 
                    my_bucket.upload_file(os.path.join(path, file), directory_name+'/'+file)
                    #dictionary['key_org'] = directory_name+'/'+file
                    #result.append(dictionary)
            t6 = time.time()
            print("FINAL UPLOAD TIME", t6-t5)
            print("Upload Successful")
            print("FINAL UPLOAD TIME", t6-t1)
            return jsonify({
                "success" : True,
                "message" : "Processing Done",
                "images" : result
                
            })
            
        except FileNotFoundError:
            print("The file was not found")

        return "Processing Done"
    app.run(host="0.0.0.0", port=5000,debug=True)