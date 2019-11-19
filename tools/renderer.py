# Render offscreen -- make sure to set the PyOpenGL platform
import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyrender


def getCameraPose(override=None):
    if not override:
        prob1 = np.random.rand(1) + 0.3
        prob2 = np.random.rand(1) + 0.3
        prob3 = np.random.rand(1) + 0.3
        # print(prob1)
        # print(prob2)
        # print(prob3)
        enable_rotx = int(np.round(prob1))
        enable_roty = int(np.round(prob2))
        enable_rotz = int(np.round(prob3))
    else:
        enable_rotx = override[0]
        enable_roty = override[1]
        enable_rotz = override[2]

    if enable_rotx:
        angle = np.random.rand(1)[0] * np.pi
        # print(angle)
        rotx = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0, 0.0],
            [-np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    if enable_roty:
        angle = np.random.rand(1)[0] * np.pi
        # print(angle)
        roty = np.array([
            [-np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    if enable_rotz:
        angle = np.random.rand(1)[0] * np.pi
        # print(angle)
        rotz = np.array([
            [np.cos(angle), np.sin(angle), 0.0, 0.0],
            [-np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    transl = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # print(enable_rotx, enable_roty, enable_rotz)

    if (not enable_rotx) & (not enable_roty) & (not enable_rotz):
        camera_pose = np.matmul(np.eye(4, 4), transl)

    if (enable_rotx) & (not enable_roty) & (not enable_rotz):
        camera_pose = np.matmul(rotx, transl)

    if (not enable_rotx) & (enable_roty) & (not enable_rotz):
        camera_pose = np.matmul(roty, transl)

    if (not enable_rotx) & (not enable_roty) & (enable_rotz):
        camera_pose = np.matmul(rotz, transl)

    if (enable_rotx) & (enable_roty) & (not enable_rotz):
        camera_pose = np.matmul(rotx, roty)
        camera_pose = np.matmul(camera_pose, transl)

    if (enable_rotx) & (enable_roty) & (enable_rotz):
        camera_pose = np.matmul(rotx, roty)
        camera_pose = np.matmul(camera_pose, rotz)
        camera_pose = np.matmul(camera_pose, transl)

    if (not enable_rotx) & (enable_roty) & (enable_rotz):
        camera_pose = np.matmul(roty, rotz)
        camera_pose = np.matmul(camera_pose, transl)

    if (enable_rotx) & (not enable_roty) & (enable_rotz):
        camera_pose = np.matmul(rotx, rotz)
        camera_pose = np.matmul(camera_pose, transl)

    return camera_pose


def render_depth(path, override=None):
    # Load the FUZE bottle trimesh and put it in a scene
    fuze_trimesh = trimesh.load(path)

    fuze_trimesh.vertices = (fuze_trimesh.vertices - np.amin(fuze_trimesh.vertices, axis=0)) / (
                np.amax(fuze_trimesh.vertices, axis=0) - np.amin(fuze_trimesh.vertices, axis=0) + 0.001) - 0.5


    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0, znear=0.15)


    camera_pose = getCameraPose(override)

    scene.add(camera, pose=camera_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(200, 200)
    depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    depth = depth / (np.max(depth) + 0.0001)

    return depth

# image = render_depth('/media/SSD/DATA/alex/ModelNet40/dresser/test/dresser_0278.off')
# plt.imshow(image)
# plt.show()