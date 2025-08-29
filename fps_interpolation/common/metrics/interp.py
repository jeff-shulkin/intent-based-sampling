from timelens.timelens.common.event import EventSequence
import numpy as np
import cv2
from typing import List

"""
Generate new frames based on reference frames, event stream, and target FPS.

Keyword arguments:
reference_frame -- the system's current RGB frame. Forms the basis for interpolation.
ref_frame_timestamp -- the reference frame's start time. Basis for filtering event stream
curr_events -- events generated starting from the reference_frame's timestamp.
model -- RNN model used to interpolate events. Takes in both reference frame and RGB events
native_FPS -- the current RGB FPS that the camera is operating at.
target_FPS -- the FPS desired for the output stream. Dictates how many frames that the system needs to generate.

Returns: List of interpolated RGB frames.
"""
def interpolate(reference_frame: np.typing.NDArray, ref_frame_timestamp: float, curr_events: EventSequence, model, native_FPS: int, target_FPS: int) -> List[np.typing.NDArray]:
    assert(native_FPS <= target_FPS, "Native FPS is greater than target FPS. Halting Program")

    # Calculate the number of frames we need to generate
    FPS_gain = target_FPS / native_FPS
    generated_frames_per_gap = FPS_gain - 1
    frame_duration = float(1 / target_FPS)
    
    generated_frames = []
    for frame_num in range(generated_frames_per_gap):
        # Obtain generated frame timestamp's 
        gen_timestamp = ref_frame_timestamp + (frame_num * frame_duration)
        frame_specific_events = curr_events.filter_by_timestamp(start_time=gen_timestamp, duration=frame_duration, make_deep_copy=False)
        generated_frames.append(generate_frame(ref_frame=reference_frame, ref_events=frame_specific_events))

    return generated_frames
    

"""
Generate a new frame given an RGB reference frame and relevant events.

Keyword arguments:
ref_frame: reference RGB frame. Can be true RGB image or generated image for continuity.
ref_events: event sequence synchronized with the ref_frame.
model: ML model used for generating new frame.

Returns: generated RGB frame.
"""
def generate_frame(ref_frame: np.typing.NDArray, ref_events: EventSequence, model):
    pass