import skvideo.io
from c3d import C3D
from keras.models import Model
from sports1M_utils import preprocess_input, decode_predictions

base_model = C3D(weights='sports1M')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

vid_path = 'homerun.mp4'
vid = skvideo.io.vread(vid_path)
# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)

features = model.predict(x)
