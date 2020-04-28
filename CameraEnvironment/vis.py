import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.animation import FuncAnimation


def visualize_run(environment,cam_traj,obj_traj,video_filename=None):
    fig, ax = pyplot.subplots()

    frame_rect_x = environment.frame_size[0] / environment.field_w
    frame_rect_y = environment.frame_size[1] / environment.field_h

    sim_points, = ax.plot([], [], 'b-')
    cam_xline = ax.axhline(y=0, c="r", linewidth=1)
    cam_yline = ax.axvline(x=0, c="r", linewidth=1)
    cam_rect = pyplot.Rectangle((0, 0), frame_rect_x*2, frame_rect_y*2, ec="r", fill=False)
    ax.add_patch(cam_rect)

    sim_xdata, sim_ydata = [], []

    text_mask = "Object inside frame: {:.2%}"
    num_in_frame = [0.0]
    text = ax.set_xlabel(text_mask.format(0.0),fontsize=20,fontweight="bold")

    def init():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return sim_points, cam_xline, cam_yline, cam_rect, text

    def update(frame):
        obj = obj_traj[frame]
        cam = cam_traj[frame]
        object_cam_dist = np.abs(cam - obj)
        num_in_frame[0] += np.all(object_cam_dist[0]<frame_rect_x and object_cam_dist[1]<frame_rect_y).astype(np.float)
        text.set_text(text_mask.format(num_in_frame[0]/(frame+1)))
        sim_xdata.append(obj[0])
        sim_ydata.append(obj[1])
        sim_points.set_data(sim_xdata, sim_ydata)
        cam_rect.set_xy((cam[0] - frame_rect_x, cam[1] - frame_rect_y))
        cam_xline.set_data([0, environment.field_w], [cam[1], cam[1]])
        cam_yline.set_data([cam[0], cam[0]], [0, environment.field_h])
        return sim_points, cam_rect, cam_xline, cam_yline

    ani = FuncAnimation(fig, update, frames=np.arange(obj_traj.shape[0]), init_func=init, blit=True, interval=2,repeat=False)

    if video_filename is None:
        pyplot.show()
    else:
        print("Saving videofile")
        ani.save(video_filename,dpi=300,progress_callback=lambda i, n: print(f'\rSaving frame {i} of {n}',end=""))
        print()
    pyplot.close(fig)
