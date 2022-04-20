from moviepy.editor import VideoFileClip
import sys
import utilis
    
video_output1 = sys.argv[2]
video_input1 = VideoFileClip(sys.argv[1])

processed_video = video_input1.fl_image(utilis.video_pipline)
processed_video.write_videofile(video_output1, audio=False)

video_input1.reader.close()
video_input1.audio.reader.close_proc()    
    
    
        
       
    
    
 
