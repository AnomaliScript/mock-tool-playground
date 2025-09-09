from pupil_apriltags import Detector
LAST_SEEN_LIMIT = 500
detector = Detector(
                        families="tag25h9",
                        nthreads=4,            # bump to 4 if available
                        quad_decimate=2.0,     # 1.0 (quality) → 1.5/2.0 (speed)
                        quad_sigma=0.0,
                        refine_edges=False,
                        decode_sharpening=0.25
                    )