import numpy as np
import  os
from vis import visualize_run
from camera_environment import CameraEnvironment

SEEDTORUN = 5
FIlENAMES = [f"test_run_{i}_seed_{j}.npz" for j in range(SEEDTORUN) for i in range(10)]

environment = CameraEnvironment()

for fn in FIlENAMES:
    file = np.load(fn)
    visualize_run(environment, file["camera_pos"], file["object_pos"], video_filename= f"{os.path.basename(fn)}.avi")
