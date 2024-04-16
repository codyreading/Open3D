import pdb
import traceback
import open3d # 0.16.0
import numpy as np

print ('\n =============================== ')
print (f' ======  Open3D=={open3d.__version__}  ======= ')
print (' =============================== \n')

if __name__ == "__main__":


    param = open3d.io.read_pinhole_camera_parameters("viewpoint.json")

    # Step 0 - Init
    WIDTH = param.intrinsic.width
    HEIGHT = param.intrinsic.height

    # Step 1 - Get scene objects
    meshFrame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Step 2 - Create visualizer object
    vizualizer = open3d.visualization.Visualizer()
    vizualizer.create_window()
    vizualizer.create_window(width=WIDTH, height=HEIGHT)

    # Step 3 - Add objects to visualizer
    #vizualizer.add_geometry(sphere1)
    vizualizer.add_geometry(meshFrame)
    

    # Step 4 - Get camera lines
    cameraLines = open3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, 
                                                                      view_height_px=HEIGHT, 
                                                                      intrinsic=param.intrinsic.intrinsic_matrix, 
                                                                      extrinsic=param.extrinsic)
    vizualizer.add_geometry(cameraLines)

    # Step 5 - Run visualizer
    vizualizer.run()
    
