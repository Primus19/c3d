import skvideo.io
from c3d import C3D
from keras.models import Model
from sports1M_utils import preprocess_input, decode_predictions
VIDEO_PATH='cat.mp4'

base_model = C3D(weights='sports1M')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

vid_path = C:\Users\primu\Downloads\Anomaly-Videos-Part-1.zip\Anomaly-Videos-Part-1\Arrest
vid = skvideo.io.vread(vid_path)
# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)

features = model.predict(x)
print(features)
