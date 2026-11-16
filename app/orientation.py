"""Display-orient decoded pixels and retain reversible encoded-pixel coordinates."""
import numpy as np


def orient_bgr(image,rotation_ccw=0):
    height,width=image.shape[:2]
    angle=float(rotation_ccw or 0)
    if not np.isfinite(angle) or abs(angle/90-round(angle/90))>1e-4:
        raise ValueError('unsupported_non_right_angle_video_rotation')
    turns=round(angle/90)%4
    matrices=(
        [[1,0,0],[0,1,0],[0,0,1]],
        [[0,1,0],[-1,0,width],[0,0,1]],
        [[-1,0,width],[0,-1,height],[0,0,1]],
        [[0,-1,height],[1,0,0],[0,0,1]],
    )
    matrix=np.asarray(matrices[turns],dtype=float)
    pixels=np.ascontiguousarray(np.rot90(image,turns)) if turns else image
    return pixels,{
        'coordinate_space':'display_oriented_original_pixels',
        'library_restores':True,
        'encoded_source_width':width,'encoded_source_height':height,
        'display_rotation_ccw_degrees':angle,
        'encoded_to_display_matrix':matrix.tolist(),
        'display_to_encoded_matrix':np.linalg.inv(matrix).tolist(),
        'received_timestamp_stage':'decoded_frame_ready',
    }


def decode_bgr(frame):
    return orient_bgr(frame.to_ndarray(format='bgr24'),getattr(frame,'rotation',0))
