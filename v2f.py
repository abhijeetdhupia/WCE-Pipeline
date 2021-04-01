import av
import os
import time
import math
import concurrent.futures


class GetFrames:
    def __init__(self, path_to_video, path_to_frames, fps=3):

        """ 
        :type path_to_video: string
        :param path_to_video: opening location of video

        :type path_to_frames: string
        :param path_to_frames: saving path of frames

        :type fps: int
        :param fps: specify frame rate for video to 
                    frame conversion

        :raises: None

        :rtype: None
        """

        self.videopath = path_to_video
        self.container = av.open(self.videopath)
        self.container.streams.video[0].thread_type = "AUTO"
        self.save_frames_path = path_to_frames
        self.frame = []

    def get_frame_names(self):
        for packet in self.container.demux():
            for frame in packet.decode():
                self.frame.append(frame)

    def frame_names(self):
        return self.frame

    def save_frames(self, frame):
        frame.to_image().save((os.path.join(self.save_frames_path, "frame-%04d.png" % frame.index)))
    
    def subfolders(self):
        self.img_list = os.listdir(self.save_frames_path)
        self.num_img = len(self.img_list)
        self.nummm = math.ceil(self.num_img/4)
        # print(self.nummm)
        os.chdir(self.save_frames_path)
        os.system(f'i=0; for f in *.png; do d=dir_$(printf %d $((i/{self.nummm}+1))); mkdir -p $d; mv "$f" $d; i=$((i+1)); done')  
        
    def __str__(self):
        return "self.container {}".format(self.container)


class GetVideo:
    def __init__(self, path_to_frames, save_video_to, desired_fps=1):
        self.path_to_frames = path_to_frames
        self.save_video_to = save_video_to + "key_frame.avi"
        self.fps = desired_fps

    def get_video(self):
        os.system(
            f"ffmpeg -r {self.fps} -f image2 -i {self.path_to_frames}%'*.png' {self.save_video_to}"
        )
