# Augmented Reality using AprilTags
In this project, I recovered the camera pose with two different approaches: PnP with coplanar assumption, and P3P followed by Procrustes. After I recovered the camera pose, I placed a virtual camera at that same position to render a virtual bird as if it stands on the tag.

## PnP with Coplanar assumption
This method recovers camera pose from 2D-3D correspondence, which needs at least 3 points. Here we get four points from AprilTags. Recall that (u v 1)' = (R T)(x y z 1)', we can compute the homogenous matrix K from correspondence and then recover R, T from K.

![](bird_collineation.gif)

## P3P
This method goes through the process of calculating the camera pose by first calculating the 3D coordinates of any 3 (out of 4) corners of the AprilTag in the camera frame. I already know the 3D coordinates of the same points in the world frame, so I will use this correspondence to solve for Camera Pose R, t in the world frame by implementing Procrustes.

![](bird_P3P.gif)
